/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl/aff.h>
#include <isl/ast.h>

#include "cuda_common.h"
#include "cuda.h"
#include "gpu.h"
#include "gpu_print.h"
#include "print.h"
#include "util.h"

static __isl_give isl_printer *print_cuda_macros(__isl_take isl_printer *p)
{
	const char *macros =
		"#define cudaCheckReturn(ret) \\\n"
		"  do { \\\n"
		"    cudaError_t cudaCheckReturn_e = (ret); \\\n"
		"    if (cudaCheckReturn_e != cudaSuccess) { \\\n"
		"      fprintf(stderr, \"CUDA error: %s\\n\", "
		"cudaGetErrorString(cudaCheckReturn_e)); \\\n"
		"      fflush(stderr); \\\n"
		"    } \\\n"
		"    assert(cudaCheckReturn_e == cudaSuccess); \\\n"
		"  } while(0)\n"
		"#define cudaCheckKernel() \\\n"
		"  do { \\\n"
		"    cudaCheckReturn(cudaGetLastError()); \\\n"
		"  } while(0)\n\n";

	p = isl_printer_print_str(p, macros);
	return p;
}

/* Print a declaration for the device array corresponding to "array" on "p".
 */
static __isl_give isl_printer *declare_device_array(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
	int i;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, " ");
	if (!array->linearize && array->n_index > 1)
		p = isl_printer_print_str(p, "(");
	p = isl_printer_print_str(p, "*dev_");
	p = isl_printer_print_str(p, array->name);
	if (!array->linearize && array->n_index > 1) {
		p = isl_printer_print_str(p, ")");
		for (i = 1; i < array->n_index; i++) {
			isl_ast_expr *bound;
			bound = isl_ast_expr_get_op_arg(array->bound_expr,
							1 + i);
			p = isl_printer_print_str(p, "[");
			p = isl_printer_print_ast_expr(p, bound);
			p = isl_printer_print_str(p, "]");
			isl_ast_expr_free(bound);
		}
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *declare_device_arrays(__isl_take isl_printer *p,
	struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		if (!gpu_array_requires_device_allocation(&prog->array[i]))
			continue;

		p = declare_device_array(p, &prog->array[i]);
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

static __isl_give isl_printer *allocate_device_arrays(
	__isl_take isl_printer *p, struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		struct gpu_array_info *array = &prog->array[i];

		if (!gpu_array_requires_device_allocation(&prog->array[i]))
			continue;
		p = ppcg_ast_expr_print_macros(array->bound_expr, p);
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p,
			"cudaCheckReturn(cudaMalloc((void **) &dev_");
		p = isl_printer_print_str(p, prog->array[i].name);
		p = isl_printer_print_str(p, ", ");
		p = gpu_array_info_print_size(p, &prog->array[i]);
		p = isl_printer_print_str(p, "));");
		p = isl_printer_end_line(p);
	}
	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);
	return p;
}

static __isl_give isl_printer *free_device_arrays(__isl_take isl_printer *p,
	struct gpu_prog *prog)
{
	int i;

	for (i = 0; i < prog->n_array; ++i) {
		if (!gpu_array_requires_device_allocation(&prog->array[i]))
			continue;
		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, "cudaCheckReturn(cudaFree(dev_");
		p = isl_printer_print_str(p, prog->array[i].name);
		p = isl_printer_print_str(p, "));");
		p = isl_printer_end_line(p);
	}

	return p;
}

/* Print code to "p" for copying "array" from the host to the device
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * gpu_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_to_device(__isl_take isl_printer *p,
	struct gpu_array_info *array)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaMemcpy(dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");

	if (gpu_array_is_scalar(array))
		p = isl_printer_print_str(p, "&");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");

	p = gpu_array_info_print_size(p, array);
	p = isl_printer_print_str(p, ", cudaMemcpyHostToDevice));");
	p = isl_printer_end_line(p);

	return p;
}

/* Print code to "p" for copying "array" back from the device to the host
 * in its entirety.  The bounds on the extent of "array" have
 * been precomputed in extract_array_info and are used in
 * gpu_array_info_print_size.
 */
static __isl_give isl_printer *copy_array_from_device(
	__isl_take isl_printer *p, struct gpu_array_info *array)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckReturn(cudaMemcpy(");
	if (gpu_array_is_scalar(array))
		p = isl_printer_print_str(p, "&");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", dev_");
	p = isl_printer_print_str(p, array->name);
	p = isl_printer_print_str(p, ", ");
	p = gpu_array_info_print_size(p, array);
	p = isl_printer_print_str(p, ", cudaMemcpyDeviceToHost));");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer* print_reverse_list(__isl_take isl_printer *p, int len, int *list)
{
	int i;

	if (len == 0)
		return p;

	p = isl_printer_print_str(p, "(");
	for (i = 0; i < len; ++i) {
		if (i)
			p = isl_printer_print_str(p, ", ");
		p = isl_printer_print_int(p, list[len - 1 - i]);
	}
	return isl_printer_print_str(p, ")");
}

/* Print the effective grid size as a list of the sizes in each
 * dimension, from innermost to outermost.
 */
static __isl_give isl_printer *print_grid_size(__isl_take isl_printer *p,
	struct ppcg_kernel *kernel)
{
	int i;
	int dim;

	dim = isl_multi_pw_aff_dim(kernel->grid_size, isl_dim_set);
	if (dim == 0)
		return p;

	p = isl_printer_print_str(p, "(");
	for (i = dim - 1; i >= 0; --i) {
		isl_ast_expr *bound;

		bound = isl_ast_expr_get_op_arg(kernel->grid_size_expr, 1 + i);
		p = isl_printer_print_ast_expr(p, bound);
		isl_ast_expr_free(bound);

		if (i > 0)
			p = isl_printer_print_str(p, ", ");
	}

	p = isl_printer_print_str(p, ")");

	return p;
}

/* Print the grid definition.
 */
static __isl_give isl_printer *print_grid(__isl_take isl_printer *p,
	struct ppcg_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "dim3 k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimGrid");
	p = print_grid_size(p, kernel);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

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
static __isl_give isl_printer *print_kernel_arguments(__isl_take isl_printer *p,
	struct gpu_prog *prog, struct ppcg_kernel *kernel, int types)
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
				&prog->array[i], NULL);
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
static __isl_give isl_printer *print_kernel_header(__isl_take isl_printer *p,
	struct gpu_prog *prog, struct ppcg_kernel *kernel)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__global__ void kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "(");
	p = print_kernel_arguments(p, prog, kernel, 1);
	p = isl_printer_print_str(p, ")");

	return p;
}

/* Print the header of the given kernel to both gen->cuda.kernel_h
 * and gen->cuda.kernel_c.
 */
static void print_kernel_headers(struct gpu_prog *prog,
	struct ppcg_kernel *kernel, struct cuda_info *cuda)
{
	isl_printer *p;

	p = isl_printer_to_file(prog->ctx, cuda->kernel_h);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);
	isl_printer_free(p);

	p = isl_printer_to_file(prog->ctx, cuda->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = print_kernel_header(p, prog, kernel);
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

static void print_indent(FILE *dst, int indent)
{
	fprintf(dst, "%*s", indent, "");
}

/* Print a list of iterators of type "type" with names "ids" to "out".
 * Each iterator is assigned one of the cuda identifiers in cuda_dims.
 * In particular, the last iterator is assigned the x identifier
 * (the first in the list of cuda identifiers).
 */
static void print_iterators(FILE *out, const char *type,
	__isl_keep isl_id_list *ids, const char *cuda_dims[])
{
	int i, n;

	n = isl_id_list_n_id(ids);
	if (n <= 0)
		return;
	print_indent(out, 4);
	fprintf(out, "%s ", type);
	for (i = 0; i < n; ++i) {
		isl_id *id;

		if (i)
			fprintf(out, ", ");
		id = isl_id_list_get_id(ids, i);
		fprintf(out, "%s = %s", isl_id_get_name(id),
			cuda_dims[n - 1 - i]);
		isl_id_free(id);
	}
	fprintf(out, ";\n");
}

static void print_kernel_iterators(FILE *out, struct ppcg_kernel *kernel)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	const char *type;
	const char *block_dims[] = { "blockIdx.x", "blockIdx.y" };
	const char *thread_dims[] = { "threadIdx.x", "threadIdx.y",
					"threadIdx.z" };

	type = isl_options_get_ast_iterator_type(ctx);

	print_iterators(out, type, kernel->block_ids, block_dims);
	print_iterators(out, type, kernel->thread_ids, thread_dims);
}

static __isl_give isl_printer *print_kernel_var(__isl_take isl_printer *p,
	struct ppcg_kernel_var *var)
{
	int j;

	p = isl_printer_start_line(p);
	if (var->type == ppcg_access_shared)
		p = isl_printer_print_str(p, "__shared__ ");
	p = isl_printer_print_str(p, var->array->type);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p,  var->name);
	for (j = 0; j < var->array->n_index; ++j) {
		isl_val *v;

		p = isl_printer_print_str(p, "[");
		v = isl_vec_get_element_val(var->size, j);
		p = isl_printer_print_val(p, v);
		isl_val_free(v);
		p = isl_printer_print_str(p, "]");
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

static __isl_give isl_printer *print_kernel_vars(__isl_take isl_printer *p,
	struct ppcg_kernel *kernel)
{
	int i;

	for (i = 0; i < kernel->n_var; ++i)
		p = print_kernel_var(p, &kernel->var[i]);

	return p;
}

/* Print a sync statement.
 */
static __isl_give isl_printer *print_sync(__isl_take isl_printer *p,
	struct ppcg_kernel_stmt *stmt)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "__syncthreads();");
	p = isl_printer_end_line(p);

	return p;
}

/* This function is called for each user statement in the AST,
 * i.e., for each kernel body statement, copy statement or sync statement.
 */
static __isl_give isl_printer *print_kernel_stmt(__isl_take isl_printer *p,
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
		return print_sync(p, stmt);
	case ppcg_kernel_domain:
		return ppcg_kernel_print_domain(p, stmt);
	}

	return p;
}

static void print_kernel(struct gpu_prog *prog, struct ppcg_kernel *kernel,
	struct cuda_info *cuda)
{
	isl_ctx *ctx = isl_ast_node_get_ctx(kernel->tree);
	isl_ast_print_options *print_options;
	isl_printer *p;

	print_kernel_headers(prog, kernel, cuda);
	fprintf(cuda->kernel_c, "{\n");
	print_kernel_iterators(cuda->kernel_c, kernel);

	p = isl_printer_to_file(ctx, cuda->kernel_c);
	p = isl_printer_set_output_format(p, ISL_FORMAT_C);
	p = isl_printer_indent(p, 4);

	p = print_kernel_vars(p, kernel);
	p = isl_printer_end_line(p);
	p = ppcg_set_macro_names(p);
	p = gpu_print_macros(p, kernel->tree);

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
							&print_kernel_stmt, NULL);
	p = isl_ast_node_print(kernel->tree, p, print_options);
	isl_printer_free(p);

	fprintf(cuda->kernel_c, "}\n");
}

/* Print code for initializing the device for execution of the transformed
 * code.  This includes declaring locally defined variables as well as
 * declaring and allocating the required copies of arrays on the device.
 */
static __isl_give isl_printer *init_device(__isl_take isl_printer *p,
	struct gpu_prog *prog)
{
	p = print_cuda_macros(p);

	p = gpu_print_local_declarations(p, prog);
	p = declare_device_arrays(p, prog);
	p = allocate_device_arrays(p, prog);

	return p;
}

/* Print code for clearing the device after execution of the transformed code.
 * In particular, free the memory that was allocated on the device.
 */
static __isl_give isl_printer *clear_device(__isl_take isl_printer *p,
	struct gpu_prog *prog)
{
	p = free_device_arrays(p, prog);

	return p;
}

/* Print a statement for copying an array to or from the device,
 * or for initializing or clearing the device.
 * The statement identifier of a copying node is called
 * "to_device_<array name>" or "from_device_<array name>" and
 * its user pointer points to the gpu_array_info of the array
 * that needs to be copied.
 * The node for initializing the device is called "init_device".
 * The node for clearing the device is called "clear_device".
 *
 * Extract the array (if any) from the identifier and call
 * init_device, clear_device, copy_array_to_device or copy_array_from_device.
 */
static __isl_give isl_printer *print_device_node(__isl_take isl_printer *p,
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
		return isl_printer_free(p);
	if (!strcmp(name, "init_device"))
		return init_device(p, prog);
	if (!strcmp(name, "clear_device"))
		return clear_device(p, prog);
	if (!array)
		return isl_printer_free(p);

	if (!prefixcmp(name, "to_device"))
		return copy_array_to_device(p, array);
	else
		return copy_array_from_device(p, array);
}

struct print_host_user_data {
	struct cuda_info *cuda;
	struct gpu_prog *prog;
};

/* Print the user statement of the host code to "p".
 *
 * The host code may contain original user statements, kernel launches,
 * statements that copy data to/from the device and statements
 * the initialize or clear the device.
 * The original user statements and the kernel launches have
 * an associated annotation, while the other statements do not.
 * The latter are handled by print_device_node.
 * The annotation on the user statements is called "user".
 *
 * In case of a kernel launch, print a block of statements that
 * defines the grid and the block and then launches the kernel.
 */
__isl_give isl_printer *print_host_user(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	int is_user;
	struct ppcg_kernel *kernel;
	struct ppcg_kernel_stmt *stmt;
	struct print_host_user_data *data;

	isl_ast_print_options_free(print_options);

	data = (struct print_host_user_data *) user;

	id = isl_ast_node_get_annotation(node);
	if (!id)
		return print_device_node(p, node, data->prog);

	is_user = !strcmp(isl_id_get_name(id), "user");
	kernel = is_user ? NULL : isl_id_get_user(id);
	stmt = is_user ? isl_id_get_user(id) : NULL;
	isl_id_free(id);

	if (is_user)
		return ppcg_kernel_print_domain(p, stmt);

	p = ppcg_start_block(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "dim3 k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimBlock");
	p = print_reverse_list(p, kernel->n_block, kernel->block_dim);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	p = print_grid(p, kernel);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "kernel");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, " <<<k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimGrid, k");
	p = isl_printer_print_int(p, kernel->id);
	p = isl_printer_print_str(p, "_dimBlock>>> (");
	p = print_kernel_arguments(p, data->prog, kernel, 0);
	p = isl_printer_print_str(p, ");");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "cudaCheckKernel();");
	p = isl_printer_end_line(p);

	p = ppcg_end_block(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

#if 0
	print_kernel(data->prog, kernel, data->cuda);
#endif

	return p;
}

static __isl_give isl_printer *print_host_code(__isl_take isl_printer *p,
	struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
	struct cuda_info *cuda)
{
	isl_ast_print_options *print_options;
	isl_ctx *ctx = isl_ast_node_get_ctx(tree);
	struct print_host_user_data data = { cuda, prog };

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
						&print_host_user, &data);

	p = gpu_print_macros(p, tree);
	p = isl_ast_node_print(tree, p, print_options);

	return p;
}

/* Given a gpu_prog "prog" and the corresponding transformed AST
 * "tree", print the entire CUDA code to "p".
 * "types" collects the types for which a definition has already
 * been printed.
 */
static __isl_give isl_printer *print_cuda(__isl_take isl_printer *p,
	struct gpu_prog *prog, __isl_keep isl_ast_node *tree,
	struct gpu_types *types, void *user)
{
	struct cuda_info *cuda = user;
	isl_printer *kernel;

	kernel = isl_printer_to_file(isl_printer_get_ctx(p), cuda->kernel_c);
	kernel = isl_printer_set_output_format(kernel, ISL_FORMAT_C);
	kernel = gpu_print_types(kernel, types, prog);
	isl_printer_free(kernel);

	if (!kernel)
		return isl_printer_free(p);

	p = print_host_code(p, prog, tree, cuda);

	return p;
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding CUDA code.
 * The names of the output files are derived from "input".
 *
 * We let generate_gpu do all the hard work and then let it call
 * us back for printing the AST in print_cuda.
 *
 * To prepare for this printing, we first open the output files
 * and we close them after generate_gpu has finished.
 */
int generate_cuda(isl_ctx *ctx, struct ppcg_options *options,
	const char *input)
{
	struct cuda_info cuda;
	int r;

	cuda_open_files(&cuda, input);

	r = generate_gpu(ctx, input, cuda.host_c, options, &print_cuda, &cuda);

	cuda_close_files(&cuda);

	return r;
}
