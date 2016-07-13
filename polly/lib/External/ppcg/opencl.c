/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege and Riyadh Baghdadi,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <ctype.h>
#include <limits.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/ast.h>

#include "opencl.h"
#include "gpu_print.h"
#include "gpu.h"
#include "ppcg.h"
#include "print.h"
#include "schedule.h"
#include "util.h"

#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))

/* options are the global options passed to generate_opencl.
 * input is the name of the input file.
 * output is the user-specified output file name and may be NULL
 *	if not specified by the user.
 * kernel_c_name is the name of the kernel_c file.
 * kprinter is an isl_printer for the kernel file.
 * host_c is the generated source file for the host code.  kernel_c is
 * the generated source file for the kernel.
 */
struct opencl_info {
	struct ppcg_options *options;
	const char *input;
	const char *output;
	char kernel_c_name[PATH_MAX];

	isl_printer *kprinter;

	FILE *host_c;
	FILE *kernel_c;
};

/* Open the file called "name" for writing or print an error message.
 */
static FILE *open_or_croak(const char *name)
{
	FILE *file;

	file = fopen(name, "w");
	if (!file)
		fprintf(stderr, "Failed to open \"%s\" for writing\n", name);
	return file;
}

/* Open the host .c file and the kernel .h and .cl files for writing.
 * Their names are derived from info->output (or info->input if
 * the user did not specify an output file name).
 * Add the necessary includes to these files, including those specified
 * by the user.
 *
 * Return 0 on success and -1 on failure.
 */
static int opencl_open_files(struct opencl_info *info)
{
	char name[PATH_MAX];
	int i;
	int len;

	if (info->output) {
		const char *ext;

		ext = strrchr(info->output, '.');
		len = ext ? ext - info->output : strlen(info->output);
		memcpy(name, info->output, len);

		info->host_c = open_or_croak(info->output);
	} else {
		len = ppcg_extract_base_name(name, info->input);

		strcpy(name + len, "_host.c");
		info->host_c = open_or_croak(name);
	}

	memcpy(info->kernel_c_name, name, len);
	strcpy(info->kernel_c_name + len, "_kernel.cl");
	info->kernel_c = open_or_croak(info->kernel_c_name);

	if (!info->host_c || !info->kernel_c)
		return -1;

	fprintf(info->host_c, "#include <assert.h>\n");
	fprintf(info->host_c, "#include <stdio.h>\n");
	fprintf(info->host_c, "#include \"ocl_utilities.h\"\n");
	if (info->options->opencl_embed_kernel_code) {
		fprintf(info->host_c, "#include \"%s\"\n\n",
			info->kernel_c_name);
	}

	for (i = 0; i < info->options->opencl_n_include_file; ++i) {
		info->kprinter = isl_printer_print_str(info->kprinter,
					"#include <");
		info->kprinter = isl_printer_print_str(info->kprinter,
					info->options->opencl_include_files[i]);
		info->kprinter = isl_printer_print_str(info->kprinter, ">\n");
	}

	return 0;
}

/* Write text to a file and escape some special characters that would break a
 * C string.
 */
static void opencl_print_escaped(const char *str, const char *end, FILE *file)
{
	const char *prev = str;

	while ((str = strpbrk(prev, "\"\\")) && str < end) {
		fwrite(prev, 1, str - prev, file);
		fprintf(file, "\\%c", *str);
		prev = str + 1;
	}

	if (*prev)
		fwrite(prev, 1, end - prev, file);
}

/* Write text to a file as a C string literal.
 *
 * This function also prints any characters after the last newline, although
 * normally the input string should end with a newline.
 */
static void opencl_print_as_c_string(const char *str, FILE *file)
{
	const char *prev = str;

	while ((str = strchr(prev, '\n'))) {
		fprintf(file, "\n\"");
		opencl_print_escaped(prev, str, file);
		fprintf(file, "\\n\"");

		prev = str + 1;
	}

	if (*prev) {
		fprintf(file, "\n\"");
		opencl_print_escaped(prev, prev + strlen(prev), file);
		fprintf(file, "\"");
	}
}

/* Write the code that we have accumulated in the kernel isl_printer to the
 * kernel.cl file.  If the opencl_embed_kernel_code option has been set, print
 * the code as a C string literal.  Start that string literal with an empty
 * line, such that line numbers reported by the OpenCL C compiler match those
 * of the kernel file.
 *
 * Return 0 on success and -1 on failure.
 */
static int opencl_write_kernel_file(struct opencl_info *opencl)
{
	char *raw = isl_printer_get_str(opencl->kprinter);

	if (!raw)
		return -1;

	if (opencl->options->opencl_embed_kernel_code) {
		fprintf(opencl->kernel_c,
			"static const char kernel_code[] = \"\\n\"");
		opencl_print_as_c_string(raw, opencl->kernel_c);
		fprintf(opencl->kernel_c, ";\n");
	} else
		fprintf(opencl->kernel_c, "%s", raw);

	free(raw);

	return 0;
}

/* Close all output files.  Write the kernel contents to the kernel file before
 * closing it.
 *
 * Return 0 on success and -1 on failure.
 */
static int opencl_close_files(struct opencl_info *info)
{
	int r = 0;

	if (info->kernel_c) {
		r = opencl_write_kernel_file(info);
		fclose(info->kernel_c);
	}
	if (info->host_c)
		fclose(info->host_c);

	return r;
}

static __isl_give isl_printer *opencl_print_host_macros(
	__isl_take isl_printer *p)
{
	const char *macros =
		"#define openclCheckReturn(ret) \\\n"
		"  if (ret != CL_SUCCESS) {\\\n"
		"    fprintf(stderr, \"OpenCL error: %s\\n\", "
		"opencl_error_string(ret)); \\\n"
		"    fflush(stderr); \\\n"
		"    assert(ret == CL_SUCCESS);\\\n  }\n";

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, macros);
	p = isl_printer_end_line(p);

	p = isl_ast_op_type_print_macro(isl_ast_op_max, p);

	return p;
}

static __isl_give isl_printer *opencl_declare_device_arrays(
	__isl_take isl_printer *p, struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		if (!gpu_array_requires_device_allocation(&prog->array[i]))
			continue;
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "cl_mem dev_");
		p = isl_printer_print_str(p, prog->array[i].name);
		p = isl_printer_print_str(p, ";");
		p = isl_printer_end_line(p);
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

/* Given an array, check whether its positive size guard expression is
 * trivial.
 */
static int is_array_positive_size_guard_trivial(struct gpu_array_info *array)
{
	isl_set *guard;
	int is_trivial;

	guard = gpu_array_positive_size_guard(array);
	is_trivial = isl_set_plain_is_universe(guard);
	isl_set_free(guard);
	return is_trivial;
}

/* Allocate a device array for "array'.
 *
 * Emit a max-expression to ensure the device array can contain at least one
 * element if the array's positive size guard expression is not trivial.
 */
static __isl_give isl_printer *allocate_device_array(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
	int need_lower_bound;

	p = ppcg_start_block(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, " = clCreateBuffer(context, ");
	p = isl_printer_print_str(p, "CL_MEM_READ_WRITE, ");

	need_lower_bound = !is_array_positive_size_guard_trivial(array);
	if (need_lower_bound) {
		p = isl_printer_print_str(p, "max(sizeof(");
		p = isl_printer_print_str(p, array->type);
		p = isl_printer_print_str(p, "), ");
	}
	p = gpu_array_info_print_size(p, array);
	if (need_lower_bound)
		p = isl_printer_print_str(p, ")");

	p = isl_printer_print_str(p, ", NULL, &err);");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(err);");
	p = isl_printer_end_line(p);

	p = ppcg_end_block(p);

	return p;
}

/* Allocate accessed device arrays.
 */
static __isl_give isl_printer *opencl_allocate_device_arrays(
	__isl_take isl_printer *p, struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		struct gpu_array_info *array = &prog->array[i];

		if (!gpu_array_requires_device_allocation(array))
			continue;

		p = allocate_device_array(p, array);
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

/* Print a call to the OpenCL clSetKernelArg() function which sets
 * the arguments of the kernel.  arg_name and arg_index are the name and the
 * index of the kernel argument.  The index of the leftmost argument of
 * the kernel is 0 whereas the index of the rightmost argument of the kernel
 * is n - 1, where n is the total number of the kernel arguments.
 * read_only_scalar is a boolean that indicates whether the argument is a read
 * only scalar.
 */
static __isl_give isl_printer *opencl_set_kernel_argument(
	__isl_take isl_printer *p, int kernel_id,
	const char *arg_name, int arg_index, int read_only_scalar)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p,
		"openclCheckReturn(clSetKernelArg(kernel");
	p = isl_printer_print_int(p, kernel_id);
	p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_int(p, arg_index);
	p = isl_printer_print_str(p, ", sizeof(");

	if (read_only_scalar) {
		p = isl_printer_print_str(p, arg_name);
		p = isl_printer_print_str(p, "), &");
	} else
		p = isl_printer_print_str(p, "cl_mem), (void *) &dev_");

	p = isl_printer_print_str(p, arg_name);
	p = isl_printer_print_str(p, "));");
	p = isl_printer_end_line(p);

	return p;
}

/* Print the block sizes as a list of the sizes in each
 * dimension.
 */
static __isl_give isl_printer *opencl_print_block_sizes(
	__isl_take isl_printer *p, struct ppcg_kernel *kernel)
{
	int i;

	if (kernel->n_block > 0)
		for (i = 0; i < kernel->n_block; ++i) {
			if (i)
				p = isl_printer_print_str(p, ", ");
			p = isl_printer_print_int(p, kernel->block_dim[i]);
		}
	else
		p = isl_printer_print_str(p, "1");

	return p;
}

/* Set the arguments of the OpenCL kernel by printing a call to the OpenCL
 * clSetKernelArg() function for each kernel argument.
 */
static __isl_give isl_printer *opencl_set_kernel_arguments(
	__isl_take isl_printer *p, struct gpu_prog *prog,
	struct ppcg_kernel *kernel)
{
	int i, n, ro;
	unsigned nparam;
	isl_space *space;
	int arg_index = 0;

	for (i = 0; i < prog->n_array; ++i) {
		int required;

		required = ppcg_kernel_requires_array_argument(kernel, i);
		if (required < 0)
			return isl_printer_free(p);
		if (!required)
			continue;
		ro = gpu_array_is_read_only_scalar(&prog->array[i]);
		opencl_set_kernel_argument(p, kernel->id, prog->array[i].name,
			arg_index, ro);
		arg_index++;
	}

	space = isl_union_set_get_space(kernel->arrays);
	nparam = isl_space_dim(space, isl_dim_param);
	for (i = 0; i < nparam; ++i) {
		const char *name;

		name = isl_space_get_dim_name(space, isl_dim_param, i);
		opencl_set_kernel_argument(p, kernel->id, name, arg_index, 1);
		arg_index++;
	}
	isl_space_free(space);

	n = isl_space_dim(kernel->space, isl_dim_set);
	for (i = 0; i < n; ++i) {
		const char *name;

		name = isl_space_get_dim_name(kernel->space, isl_dim_set, i);
		opencl_set_kernel_argument(p, kernel->id, name, arg_index, 1);
		arg_index++;
	}

	return p;
}

/* Print the arguments to a kernel declaration or call.  If "types" is set,
 * then print a declaration (including the types of the arguments).
 *
 * The arguments are printed in the following order
 * - the arrays accessed by the kernel
 * - the parameters
 * - the host loop iterators
 */
static __isl_give isl_printer *opencl_print_kernel_arguments(
	__isl_take isl_printer *p, struct gpu_prog *prog,
	struct ppcg_kernel *kernel, int types)
{
	int i, n;
	int first = 1;
	unsigned nparam;
	isl_space *space;
	const char *type;

	for (i = 0; i < prog->n_array; ++i) {
		int required;

		required = ppcg_kernel_requires_array_argument(kernel, i);
		if (required < 0)
			return isl_printer_free(p);
		if (!required)
			continue;

		if (!first)
			p = isl_printer_print_str(p, ", ");

		if (types)
			p = gpu_array_info_print_declaration_argument(p,
				&prog->array[i], "__global");
		else
			p = gpu_array_info_print_call_argument(p,
				&prog->array[i]);

		first = 0;
	}

	space = isl_union_set_get_space(kernel->arrays);
	nparam = isl_space_dim(space, isl_dim_param);
	for (i = 0; i < nparam; ++i) {
		const char *name;

		name = isl_space_get_dim_name(space, isl_dim_param, i);

		if (!first)
			p = isl_printer_print_str(p, ", ");
		if (types)
			p = isl_printer_print_str(p, "int ");
		p = isl_printer_print_str(p, name);

		first = 0;
	}
	isl_space_free(space);

	n = isl_space_dim(kernel->space, isl_dim_set);
	type = isl_options_get_ast_iterator_type(prog->ctx);
	for (i = 0; i < n; ++i) {
		const char *name;

		if (!first)
			p = isl_printer_print_str(p, ", ");
		name = isl_space_get_dim_name(kernel->space, isl_dim_set, i);
		if (types) {
			p = isl_printer_print_str(p, type);
			p = isl_printer_print_str(p, " ");
		}
		p = isl_printer_print_str(p, name);

		first = 0;
	}

	return p;
}

/* Print the header of the given kernel.
 */
static __isl_give isl_printer *opencl_print_kernel_header(
	__isl_take isl_printer *p, struct gpu_prog *prog,
	struct ppcg_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__kernel void kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "(");
	p = opencl_print_kernel_arguments(p, prog, kernel, 1);
	p = isl_printer_print_str(p, ")");
	p = isl_printer_end_line(p);

	return p;
}

/* Print a list of iterators of type "type" with names "ids" to "p".
 * Each iterator is assigned the corresponding opencl identifier returned
 * by the function "opencl_id".
 * Unlike the equivalent function in the CUDA backend which prints iterators
 * in reverse order to promote coalescing, this function does not print
 * iterators in reverse order.  The OpenCL backend currently does not take
 * into account any coalescing considerations.
 */
static __isl_give isl_printer *print_iterators(__isl_take isl_printer *p,
	const char *type, __isl_keep isl_id_list *ids, const char *opencl_id)
{
	int i, n;

	n = isl_id_list_n_id(ids);
	if (n <= 0)
		return p;
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, type);
	p = isl_printer_print_str(p, " ");
	for (i = 0; i < n; ++i) {
		isl_id *id;

		if (i)
			p = isl_printer_print_str(p, ", ");
		id = isl_id_list_get_id(ids, i);
		p = isl_printer_print_id(p, id);
		isl_id_free(id);
		p = isl_printer_print_str(p, " = ");
		p = isl_printer_print_str(p, opencl_id);
		p = isl_printer_print_str(p, "(");
		p = isl_printer_print_int(p, i);
		p = isl_printer_print_str(p, ")");
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *opencl_print_kernel_iterators(
	__isl_take isl_printer *p, struct ppcg_kernel *kernel)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	const char *type;

	type = isl_options_get_ast_iterator_type(ctx);

	p = print_iterators(p, type, kernel->block_ids, "get_group_id");
	p = print_iterators(p, type, kernel->thread_ids, "get_local_id");

	return p;
}

static __isl_give isl_printer *opencl_print_kernel_var(
	__isl_take isl_printer *p, struct ppcg_kernel_var *var)
{
	int j;
	isl_val *v;

	p = isl_printer_start_line(p);
	if (var->type == ppcg_access_shared)
		p = isl_printer_print_str(p, "__local ");
	p = isl_printer_print_str(p, var->array->type);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, var->name);
	for (j = 0; j < var->array->n_index; ++j) {
		p = isl_printer_print_str(p, "[");
		v = isl_vec_get_element_val(var->size, j);
		p = isl_printer_print_val(p, v);
		p = isl_printer_print_str(p, "]");
		isl_val_free(v);
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *opencl_print_kernel_vars(
		__isl_take isl_printer *p, struct ppcg_kernel *kernel)
{
	int i;

	for (i = 0; i < kernel->n_var; ++i)
		p = opencl_print_kernel_var(p, &kernel->var[i]);

	return p;
}

/* Print a call to barrier() which is a sync statement.
 * All work-items in a work-group executing the kernel on a processor must
 * execute the barrier() function before any are allowed to continue execution
 * beyond the barrier.
 * The flag CLK_LOCAL_MEM_FENCE makes the barrier function either flush any
 * variables stored in local memory or queue a memory fence to ensure correct
 * ordering of memory operations to local memory.
 * The flag CLK_GLOBAL_MEM_FENCE makes the barrier function queue a memory
 * fence to ensure correct ordering of memory operations to global memory.
 */
static __isl_give isl_printer *opencl_print_sync(__isl_take isl_printer *p,
	struct ppcg_kernel_stmt *stmt)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p,
		"barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);");
	p = isl_printer_end_line(p);

	return p;
}

/* Data structure containing function names for which the calls
 * should be changed from
 *
 *	name(arg)
 *
 * to
 *
 *	opencl_name((type) (arg))
 */
static struct ppcg_opencl_fn {
	const char *name;
	const char *opencl_name;
	const char *type;
} opencl_fn[] = {
	{ "expf",	"exp",		"float" },
	{ "powf",	"pow",		"float" },
	{ "sqrtf",	"sqrt",		"float" },
};

#define ARRAY_SIZE(array) (sizeof(array)/sizeof(*array))

/* If the name of function called by "expr" matches any of those
 * in ppcg_opencl_fn, then replace the call by a cast to the corresponding
 * type in ppcg_opencl_fn and a call to corresponding OpenCL function.
 */
static __isl_give pet_expr *map_opencl_call(__isl_take pet_expr *expr,
	void *user)
{
	const char *name;
	int i;

	name = pet_expr_call_get_name(expr);
	for (i = 0; i < ARRAY_SIZE(opencl_fn); ++i) {
		pet_expr *arg;

		if (strcmp(name, opencl_fn[i].name))
			continue;
		expr = pet_expr_call_set_name(expr, opencl_fn[i].opencl_name);
		arg = pet_expr_get_arg(expr, 0);
		arg = pet_expr_new_cast(opencl_fn[i].type, arg);
		expr = pet_expr_set_arg(expr, 0, arg);
	}
	return expr;
}

/* Print the body of a statement from the input program,
 * for use in OpenCL code.
 *
 * Before calling ppcg_kernel_print_domain to print the actual statement body,
 * we first modify this body to take into account that the output code
 * is OpenCL code.  In particular, if the statement calls any function
 * with a "f" suffix, then it needs to be replaced by a call to
 * the corresponding function without suffix after casting the argument
 * to a float.
 */
static __isl_give isl_printer *print_opencl_kernel_domain(
	__isl_take isl_printer *p, struct ppcg_kernel_stmt *stmt)
{
	struct pet_stmt *ps;
	pet_tree *tree;

	ps = stmt->u.d.stmt->stmt;
	tree = pet_tree_copy(ps->body);
	ps->body = pet_tree_map_call_expr(ps->body, &map_opencl_call, NULL);
	p = ppcg_kernel_print_domain(p, stmt);
	pet_tree_free(ps->body);
	ps->body = tree;

	return p;
}

/* This function is called for each user statement in the AST,
 * i.e., for each kernel body statement, copy statement or sync statement.
 */
static __isl_give isl_printer *opencl_print_kernel_stmt(
	__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	struct ppcg_kernel_stmt *stmt;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	isl_ast_print_options_free(print_options);

	switch (stmt->type) {
	case ppcg_kernel_copy:
		return ppcg_kernel_print_copy(p, stmt);
	case ppcg_kernel_sync:
		return opencl_print_sync(p, stmt);
	case ppcg_kernel_domain:
		return print_opencl_kernel_domain(p, stmt);
	}

	return p;
}

/* Return true if there is a double array in prog->array or
 * if any of the types in prog->scop involve any doubles.
 * To check the latter condition, we simply search for the string "double"
 * in the type definitions, which may result in false positives.
 */
static __isl_give int any_double_elements(struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i)
		if (strcmp(prog->array[i].type, "double") == 0)
			return 1;

	for (i = 0; i < prog->scop->pet->n_type; ++i) {
		struct pet_type *type = prog->scop->pet->types[i];

		if (strstr(type->definition, "double"))
			return 1;
	}

	return 0;
}

/* Prints a #pragma to enable support for double floating-point
 * precision.  OpenCL 1.0 adds support for double precision floating-point as
 * an optional extension. An application that wants to use double will need to
 * include the #pragma OPENCL EXTENSION cl_khr_fp64 : enable directive before
 * any double precision data type is declared in the kernel code.
 */
static __isl_give isl_printer *opencl_enable_double_support(
	__isl_take isl_printer *p)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "#pragma OPENCL EXTENSION cl_khr_fp64 :"
		" enable");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *opencl_print_kernel(struct gpu_prog *prog,
	struct ppcg_kernel *kernel, __isl_take isl_printer *p)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	isl_ast_print_options *print_options;

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
				&opencl_print_kernel_stmt, NULL);

	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = opencl_print_kernel_header(p, prog, kernel);
	p = isl_printer_print_str(p, "{");
	p = isl_printer_end_line(p);
	p = isl_printer_indent(p, 4);
	p = opencl_print_kernel_iterators(p, kernel);
	p = opencl_print_kernel_vars(p, kernel);
	p = isl_printer_end_line(p);
	p = isl_ast_op_type_print_macro(isl_ast_op_fdiv_q, p);
	p = ppcg_print_macros(p, kernel->tree);
	p = isl_ast_node_print(kernel->tree, p, print_options);
	p = isl_printer_indent(p, -4);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "}");
	p = isl_printer_end_line(p);

	return p;
}

struct print_host_user_data_opencl {
	struct opencl_info *opencl;
	struct gpu_prog *prog;
};

/* This function prints the i'th block size multiplied by the i'th grid size,
 * where i (a parameter to this function) is one of the possible dimensions of
 * grid sizes and block sizes.
 * If the dimension of block sizes is not equal to the dimension of grid sizes
 * the output is calculated as follows:
 *
 * Suppose that:
 * block_sizes[dim1] is the list of blocks sizes and it contains dim1 elements.
 * grid_sizes[dim2] is the list of grid sizes and it contains dim2 elements.
 *
 * The output is:
 * If (i > dim2) then the output is block_sizes[i]
 * If (i > dim1) then the output is grid_sizes[i]
 */
static __isl_give isl_printer *opencl_print_total_number_of_work_items_for_dim(
	__isl_take isl_printer *p, struct ppcg_kernel *kernel, int i)
{
	int grid_dim, block_dim;
	isl_pw_aff *bound_grid;

	grid_dim = isl_multi_pw_aff_dim(kernel->grid_size, isl_dim_set);
	block_dim = kernel->n_block;

	if (i < min(grid_dim, block_dim)) {
		bound_grid = isl_multi_pw_aff_get_pw_aff(kernel->grid_size, i);
		p = isl_printer_print_str(p, "(");
		p = isl_printer_print_pw_aff(p, bound_grid);
		p = isl_printer_print_str(p, ") * ");
		p = isl_printer_print_int(p, kernel->block_dim[i]);
		isl_pw_aff_free(bound_grid);
	} else if (i >= grid_dim)
		p = isl_printer_print_int(p, kernel->block_dim[i]);
	else {
		bound_grid = isl_multi_pw_aff_get_pw_aff(kernel->grid_size, i);
		p = isl_printer_print_pw_aff(p, bound_grid);
		isl_pw_aff_free(bound_grid);
	}

	return p;
}

/* Print a list that represents the total number of work items.  The list is
 * constructed by performing an element-wise multiplication of the block sizes
 * and the grid sizes.  To explain how the list is constructed, suppose that:
 * block_sizes[dim1] is the list of blocks sizes and it contains dim1 elements.
 * grid_sizes[dim2] is the list of grid sizes and it contains dim2 elements.
 *
 * The output of this function is constructed as follows:
 * If (dim1 > dim2) then the output is the following list:
 * grid_sizes[0]*block_sizes[0], ..., grid_sizes[dim2-1]*block_sizes[dim2-1],
 * block_sizes[dim2], ..., block_sizes[dim1-2], block_sizes[dim1-1].
 *
 * If (dim2 > dim1) then the output is the following list:
 * grid_sizes[0]*block_sizes[0], ..., grid_sizes[dim1-1] * block_sizes[dim1-1],
 * grid_sizes[dim1], grid_sizes[dim2-2], grid_sizes[dim2-1].
 *
 * To calculate the total number of work items out of the list constructed by
 * this function, the user should multiply the elements of the list.
 */
static __isl_give isl_printer *opencl_print_total_number_of_work_items_as_list(
	__isl_take isl_printer *p, struct ppcg_kernel *kernel)
{
	int i;
	int grid_dim, block_dim;

	grid_dim = isl_multi_pw_aff_dim(kernel->grid_size, isl_dim_set);
	block_dim = kernel->n_block;

	if ((grid_dim <= 0) || (block_dim <= 0)) {
		p = isl_printer_print_str(p, "1");
		return p;
	}

	for (i = 0; i <= max(grid_dim, block_dim) - 1; i++) {
		if (i > 0)
			p = isl_printer_print_str(p, ", ");

		p = opencl_print_total_number_of_work_items_for_dim(p,
			kernel, i);
	}

	return p;
}

/* Copy "array" from the host to the device (to_host = 0) or
 * back from the device to the host (to_host = 1).
 */
static __isl_give isl_printer *copy_array(__isl_take isl_printer *p,
	struct gpu_array_info *array, int to_host)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(");
	if (to_host)
		p = isl_printer_print_str(p, "clEnqueueReadBuffer");
	else
		p = isl_printer_print_str(p, "clEnqueueWriteBuffer");
	p = isl_printer_print_str(p, "(queue, dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", CL_TRUE, 0, ");
	p = gpu_array_info_print_size(p, array);

	if (gpu_array_is_scalar(array))
		p = isl_printer_print_str(p, ", &");
	else
		p = isl_printer_print_str(p, ", ");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", 0, NULL, NULL));");
	p = isl_printer_end_line(p);

	return p;
}

/* Print a statement for copying an array to or from the device.
 * The statement identifier is called "to_device_<array name>" or
 * "from_device_<array name>" and its user pointer points
 * to the gpu_array_info of the array that needs to be copied.
 *
 * Extract the array from the identifier and call
 * copy_array_to_device or copy_array_from_device.
 */
static __isl_give isl_printer *print_to_from_device(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node, struct gpu_prog *prog)
{
	isl_ast_expr *expr, *arg;
	isl_id *id;
	const char *name;
	struct gpu_array_info *array;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	id = isl_ast_expr_get_id(arg);
	name = isl_id_get_name(id);
	array = isl_id_get_user(id);
	isl_id_free(id);
	isl_ast_expr_free(arg);
	isl_ast_expr_free(expr);

	if (!name)
		array = NULL;
	if (!array)
		return isl_printer_free(p);

	if (!prefixcmp(name, "to_device"))
		return copy_array(p, array, 0);
	else
		return copy_array(p, array, 1);
}

/* Print the user statement of the host code to "p".
 *
 * The host code may contain original user statements, kernel launches and
 * statements that copy data to/from the device.
 * The original user statements and the kernel launches have
 * an associated annotation, while the data copy statements do not.
 * The latter are handled by print_to_from_device.
 * The annotation on the user statements is called "user".
 *
 * In case of a kernel launch, print a block of statements that
 * defines the grid and the work group and then launches the kernel.
 *
 * A grid is composed of many work groups (blocks), each work group holds
 * many work-items (threads).
 *
 * global_work_size[kernel->n_block] represents the total number of work
 * items.  It points to an array of kernel->n_block unsigned
 * values that describe the total number of work-items that will execute
 * the kernel.  The total number of work-items is computed as:
 * global_work_size[0] *...* global_work_size[kernel->n_block - 1].
 *
 * The size of each work group (i.e. the number of work-items in each work
 * group) is described using block_size[kernel->n_block].  The total
 * number of work-items in a block (work-group) is computed as:
 * block_size[0] *... * block_size[kernel->n_block - 1].
 *
 * For more information check:
 * http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clEnqueueNDRangeKernel.html
 */
static __isl_give isl_printer *opencl_print_host_user(
	__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	int is_user;
	struct ppcg_kernel *kernel;
	struct ppcg_kernel_stmt *stmt;
	struct print_host_user_data_opencl *data;

	isl_ast_print_options_free(print_options);

	data = (struct print_host_user_data_opencl *) user;

	id = isl_ast_node_get_annotation(node);
	if (!id)
		return print_to_from_device(p, node, data->prog);

	is_user = !strcmp(isl_id_get_name(id), "user");
	kernel = is_user ? NULL : isl_id_get_user(id);
	stmt = is_user ? isl_id_get_user(id) : NULL;
	isl_id_free(id);

	if (is_user)
		return ppcg_kernel_print_domain(p, stmt);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "{");
	p = isl_printer_end_line(p);
	p = isl_printer_indent(p, 2);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "size_t global_work_size[");

	if (kernel->n_block > 0)
		p = isl_printer_print_int(p, kernel->n_block);
	else
		p = isl_printer_print_int(p, 1);

	p = isl_printer_print_str(p, "] = {");
	p = opencl_print_total_number_of_work_items_as_list(p, kernel);
	p = isl_printer_print_str(p, "};");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "size_t block_size[");

	if (kernel->n_block > 0)
		p = isl_printer_print_int(p, kernel->n_block);
	else
		p = isl_printer_print_int(p, 1);

	p = isl_printer_print_str(p, "] = {");
	p = opencl_print_block_sizes(p, kernel);
	p = isl_printer_print_str(p, "};");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cl_kernel kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, " = clCreateKernel(program, \"kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "\", &err);");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(err);");
	p = isl_printer_end_line(p);

	opencl_set_kernel_arguments(p, data->prog, kernel);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(clEnqueueNDRangeKernel"
		"(queue, kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, ", ");
	if (kernel->n_block > 0)
		p = isl_printer_print_int(p, kernel->n_block);
	else
		p = isl_printer_print_int(p, 1);

	p = isl_printer_print_str(p, ", NULL, global_work_size, "
					"block_size, "
					"0, NULL, NULL));");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn("
					"clReleaseKernel(kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "));");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "clFinish(queue);");
	p = isl_printer_end_line(p);
	p = isl_printer_indent(p, -2);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "}");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	data->opencl->kprinter = opencl_print_kernel(data->prog, kernel,
						data->opencl->kprinter);

	return p;
}

static __isl_give isl_printer *opencl_print_host_code(
	__isl_take isl_printer *p, struct gpu_prog *prog,
	__isl_keep isl_ast_node *tree, struct opencl_info *opencl)
{
	isl_ast_print_options *print_options;
	isl_ctx *ctx = isl_ast_node_get_ctx(tree);
	struct print_host_user_data_opencl data = { opencl, prog };

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
				&opencl_print_host_user, &data);

	p = ppcg_print_macros(p, tree);
	p = isl_ast_node_print(tree, p, print_options);

	return p;
}

/* Create an OpenCL device, context, command queue and build the kernel.
 * input is the name of the input file provided to ppcg.
 */
static __isl_give isl_printer *opencl_setup(__isl_take isl_printer *p,
	const char *input, struct opencl_info *info)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cl_device_id device;");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cl_context context;");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cl_program program;");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cl_command_queue queue;");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cl_int err;");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "device = opencl_create_device(");
	p = isl_printer_print_int(p, info->options->opencl_use_gpu);
	p = isl_printer_print_str(p, ");");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "context = clCreateContext(NULL, 1, "
		"&device, NULL, NULL, &err);");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(err);");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "queue = clCreateCommandQueue"
					"(context, device, 0, &err);");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(err);");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "program = ");

	if (info->options->opencl_embed_kernel_code) {
		p = isl_printer_print_str(p, "opencl_build_program_from_string("
						"context, device, kernel_code, "
						"sizeof(kernel_code), \"");
	} else {
		p = isl_printer_print_str(p, "opencl_build_program_from_file("
						"context, device, \"");
		p = isl_printer_print_str(p, info->kernel_c_name);
		p = isl_printer_print_str(p, "\", \"");
	}

	if (info->options->opencl_compiler_options)
		p = isl_printer_print_str(p,
					info->options->opencl_compiler_options);

	p = isl_printer_print_str(p, "\");");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *opencl_release_cl_objects(
	__isl_take isl_printer *p, struct opencl_info *info)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(clReleaseCommandQueue"
					"(queue));");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(clReleaseProgram"
					"(program));");
	p = isl_printer_end_line(p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn(clReleaseContext"
					"(context));");
	p = isl_printer_end_line(p);

	return p;
}

/* Free the device array corresponding to "array"
 */
static __isl_give isl_printer *release_device_array(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "openclCheckReturn("
					"clReleaseMemObject(dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, "));");
	p = isl_printer_end_line(p);

	return p;
}

/* Free the accessed device arrays.
 */
static __isl_give isl_printer *opencl_release_device_arrays(
	__isl_take isl_printer *p, struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		struct gpu_array_info *array = &prog->array[i];
		if (!gpu_array_requires_device_allocation(array))
			continue;

		p = release_device_array(p, array);
	}
	return p;
}

/* Given a gpu_prog "prog" and the corresponding transformed AST
 * "tree", print the entire OpenCL code to "p".
 */
static __isl_give isl_printer *print_opencl(__isl_take isl_printer *p,
	struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
	struct gpu_types *types, void *user)
{
	struct opencl_info *opencl = user;

	opencl->kprinter = isl_printer_set_output_format(opencl->kprinter,
							ISL_FORMAT_C);
	if (any_double_elements(prog))
		opencl->kprinter = opencl_enable_double_support(
							opencl->kprinter);
	if (opencl->options->opencl_print_kernel_types)
		opencl->kprinter = gpu_print_types(opencl->kprinter, types,
								prog);

	if (!opencl->kprinter)
		return isl_printer_free(p);

	p = ppcg_start_block(p);

	p = opencl_print_host_macros(p);

	p = gpu_print_local_declarations(p, prog);
	p = opencl_declare_device_arrays(p, prog);
	p = opencl_setup(p, opencl->input, opencl);
	p = opencl_allocate_device_arrays(p, prog);

	p = opencl_print_host_code(p, prog, tree, opencl);

	p = opencl_release_device_arrays(p, prog);
	p = opencl_release_cl_objects(p, opencl);

	p = ppcg_end_block(p);

	return p;
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding OpenCL code.
 * The host code is written to "output" or a name derived from
 * "input" if "output" is NULL.
 * The kernel code is placed in separate files with names
 * derived from "output" or "input".
 *
 * We let generate_gpu do all the hard work and then let it call
 * us back for printing the AST in print_opencl.
 *
 * To prepare for this printing, we first open the output files
 * and we close them after generate_gpu has finished.
 */
int generate_opencl(isl_ctx *ctx, struct ppcg_options *options,
	const char *input, const char *output)
{
	struct opencl_info opencl = { options, input, output };
	int r;

	opencl.kprinter = isl_printer_to_str(ctx);
	r = opencl_open_files(&opencl);

	if (r >= 0)
		r = generate_gpu(ctx, input, opencl.host_c, options,
				&print_opencl, &opencl);

	if (opencl_close_files(&opencl) < 0)
		r = -1;
	isl_printer_free(opencl.kprinter);

	return r;
}
