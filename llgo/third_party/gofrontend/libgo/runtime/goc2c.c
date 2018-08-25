// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
 * Translate a .goc file into a .c file.  A .goc file is a combination
 * of a limited form of Go with C.
 */

/*
	package PACKAGENAME
	{# line}
	func NAME([NAME TYPE { , NAME TYPE }]) [(NAME TYPE { , NAME TYPE })] \{
	  C code with proper brace nesting
	\}
*/

/*
 * We generate C code which implements the function such that it can
 * be called from Go and executes the C code.
 */

#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* Package path to use.  */
static const char *pkgpath;

/* Package prefix to use.  */
static const char *prefix;

/* File and line number */
static const char *file;
static unsigned int lineno = 1;

/* List of names and types.  */
struct params {
	struct params *next;
	char *name;
	char *type;
};

char *argv0;

static void
sysfatal(char *fmt, ...)
{
	char buf[256];
	va_list arg;

	va_start(arg, fmt);
	vsnprintf(buf, sizeof buf, fmt, arg);
	va_end(arg);

	fprintf(stderr, "%s: %s\n", argv0 ? argv0 : "<prog>", buf);
	exit(1);
}

/* Unexpected EOF.  */
static void
bad_eof(void)
{
	sysfatal("%s:%ud: unexpected EOF\n", file, lineno);
}

/* Out of memory.  */
static void
bad_mem(void)
{
	sysfatal("%s:%ud: out of memory\n", file, lineno);
}

/* Allocate memory without fail.  */
static void *
xmalloc(unsigned int size)
{
	void *ret = malloc(size);
	if (ret == NULL)
		bad_mem();
	return ret;
}

/* Reallocate memory without fail.  */
static void*
xrealloc(void *buf, unsigned int size)
{
	void *ret = realloc(buf, size);
	if (ret == NULL)
		bad_mem();
	return ret;
}

/* Copy a string into memory without fail.  */
static char *
xstrdup(const char *p)
{
	char *ret = xmalloc(strlen(p) + 1);
	strcpy(ret, p);
	return ret;
}

/* Free a list of parameters.  */
static void
free_params(struct params *p)
{
	while (p != NULL) {
		struct params *next;

		next = p->next;
		free(p->name);
		free(p->type);
		free(p);
		p = next;
	}
}

/* Read a character, tracking lineno.  */
static int
getchar_update_lineno(void)
{
	int c;

	c = getchar();
	if (c == '\n')
		++lineno;
	return c;
}

/* Read a character, giving an error on EOF, tracking lineno.  */
static int
getchar_no_eof(void)
{
	int c;

	c = getchar_update_lineno();
	if (c == EOF)
		bad_eof();
	return c;
}

/* Read a character, skipping comments.  */
static int
getchar_skipping_comments(void)
{
	int c;

	while (1) {
		c = getchar_update_lineno();
		if (c != '/')
			return c;

		c = getchar();
		if (c == '/') {
			do {
				c = getchar_update_lineno();
			} while (c != EOF && c != '\n');
			return c;
		} else if (c == '*') {
			while (1) {
				c = getchar_update_lineno();
				if (c == EOF)
					return EOF;
				if (c == '*') {
					do {
						c = getchar_update_lineno();
					} while (c == '*');
					if (c == '/')
						break;
				}
			}
		} else {
			ungetc(c, stdin);
			return '/';
		}
	}
}

/*
 * Read and return a token.  Tokens are string or character literals
 * or else delimited by whitespace or by [(),{}].
 * The latter are all returned as single characters.
 */
static char *
read_token(void)
{
	int c, q;
	char *buf;
	unsigned int alc, off;
	const char* delims = "(),{}";

	while (1) {
		c = getchar_skipping_comments();
		if (c == EOF)
			return NULL;
		if (!isspace(c))
			break;
	}
	alc = 16;
	buf = xmalloc(alc + 1);
	off = 0;
	if(c == '"' || c == '\'') {
		q = c;
		buf[off] = c;
		++off;
		while (1) {
			if (off+2 >= alc) { // room for c and maybe next char
				alc *= 2;
				buf = xrealloc(buf, alc + 1);
			}
			c = getchar_no_eof();
			buf[off] = c;
			++off;
			if(c == q)
				break;
			if(c == '\\') {
				buf[off] = getchar_no_eof();
				++off;
			}
		}
	} else if (strchr(delims, c) != NULL) {
		buf[off] = c;
		++off;
	} else {
		while (1) {
			if (off >= alc) {
				alc *= 2;
				buf = xrealloc(buf, alc + 1);
			}
			buf[off] = c;
			++off;
			c = getchar_skipping_comments();
			if (c == EOF)
				break;
			if (isspace(c) || strchr(delims, c) != NULL) {
				if (c == '\n')
					lineno--;
				ungetc(c, stdin);
				break;
			}
		}
	}
	buf[off] = '\0';
	return buf;
}

/* Read a token, giving an error on EOF.  */
static char *
read_token_no_eof(void)
{
	char *token = read_token();
	if (token == NULL)
		bad_eof();
	return token;
}

/* Read the package clause, and return the package name.  */
static char *
read_package(void)
{
	char *token;

	token = read_token_no_eof();
	if (token == NULL)
		sysfatal("%s:%ud: no token\n", file, lineno);
	if (strcmp(token, "package") != 0) {
		sysfatal("%s:%ud: expected \"package\", got \"%s\"\n",
			file, lineno, token);
	}
	return read_token_no_eof();
}

/* Read and copy preprocessor lines.  */
static void
read_preprocessor_lines(void)
{
	while (1) {
		int c;

		do {
			c = getchar_skipping_comments();
		} while (isspace(c));
		if (c != '#') {
			ungetc(c, stdin);
			break;
		}
		putchar(c);
		do {
			c = getchar_update_lineno();
			putchar(c);
		} while (c != '\n');
	}
}

/*
 * Read a type in Go syntax and return a type in C syntax.  We only
 * permit basic types and pointers.
 */
static char *
read_type(void)
{
	char *p, *op, *q;
	int pointer_count;
	unsigned int len;

	p = read_token_no_eof();
	if (*p != '*') {
		/* Convert the Go type "int" to the C type "intgo",
		   and similarly for "uint".  */
		if (strcmp(p, "int") == 0)
			return xstrdup("intgo");
		else if (strcmp(p, "uint") == 0)
			return xstrdup("uintgo");
		return p;
	}
	op = p;
	pointer_count = 0;
	while (*p == '*') {
		++pointer_count;
		++p;
	}

	/* Convert the Go type "int" to the C type "intgo", and
	   similarly for "uint".  */
	if (strcmp(p, "int") == 0)
	  p = (char *) "intgo";
	else if (strcmp(p, "uint") == 0)
	  p = (char *) "uintgo";

	len = strlen(p);
	q = xmalloc(len + pointer_count + 1);
	memcpy(q, p, len);
	while (pointer_count > 0) {
		q[len] = '*';
		++len;
		--pointer_count;
	}
	q[len] = '\0';
	free(op);
	return q;
}

/*
 * Read a list of parameters.  Each parameter is a name and a type.
 * The list ends with a ')'.  We have already read the '('.
 */
static struct params *
read_params()
{
	char *token;
	struct params *ret, **pp, *p;

	ret = NULL;
	pp = &ret;
	token = read_token_no_eof();
	if (strcmp(token, ")") != 0) {
		while (1) {
			p = xmalloc(sizeof(struct params));
			p->name = token;
			p->type = read_type();
			p->next = NULL;
			*pp = p;
			pp = &p->next;

			token = read_token_no_eof();
			if (strcmp(token, ",") != 0)
				break;
			token = read_token_no_eof();
		}
	}
	if (strcmp(token, ")") != 0) {
		sysfatal("%s:%ud: expected '('\n",
			file, lineno);
	}
	return ret;
}

/*
 * Read a function header.  This reads up to and including the initial
 * '{' character.  Returns 1 if it read a header, 0 at EOF.
 */
static int
read_func_header(char **name, struct params **params, struct params **rets)
{
	int lastline;
	char *token;

	lastline = -1;
	while (1) {
		token = read_token();
		if (token == NULL)
			return 0;
		if (strcmp(token, "func") == 0) {
			if(lastline != -1)
				printf("\n");
			break;
		}
		if (lastline != lineno) {
			if (lastline == lineno-1)
				printf("\n");
			else
				printf("\n#line %d \"%s\"\n", lineno, file);
			lastline = lineno;
		}
		printf("%s ", token);
	}

	*name = read_token_no_eof();

	token = read_token();
	if (token == NULL || strcmp(token, "(") != 0) {
		sysfatal("%s:%ud: expected \"(\"\n",
			file, lineno);
	}
	*params = read_params();

	token = read_token();
	if (token == NULL || strcmp(token, "(") != 0)
		*rets = NULL;
	else {
		*rets = read_params();
		token = read_token();
	}
	if (token == NULL || strcmp(token, "{") != 0) {
		sysfatal("%s:%ud: expected \"{\"\n",
			file, lineno);
	}
	return 1;
}

/* Write out parameters.  */
static void
write_params(struct params *params, int *first)
{
	struct params *p;

	for (p = params; p != NULL; p = p->next) {
		if (*first)
			*first = 0;
		else
			printf(", ");
		printf("%s %s", p->type, p->name);
	}
}

/* Define the gcc function return type if necessary.  */
static void
define_gcc_return_type(char *package, char *name, struct params *rets)
{
	struct params *p;

	if (rets == NULL || rets->next == NULL)
		return;
	printf("struct %s_%s_ret {\n", package, name);
	for (p = rets; p != NULL; p = p->next)
		printf("  %s %s;\n", p->type, p->name);
	printf("};\n");
}

/* Write out the gcc function return type.  */
static void
write_gcc_return_type(char *package, char *name, struct params *rets)
{
	if (rets == NULL)
		printf("void");
	else if (rets->next == NULL)
		printf("%s", rets->type);
	else
		printf("struct %s_%s_ret", package, name);
}

/* Write out a gcc function header.  */
static void
write_gcc_func_header(char *package, char *name, struct params *params,
		      struct params *rets)
{
	int first;
	struct params *p;

	define_gcc_return_type(package, name, rets);
	write_gcc_return_type(package, name, rets);
	printf(" %s_%s(", package, name);
	first = 1;
	write_params(params, &first);
	printf(") __asm__ (GOSYM_PREFIX \"");
	if (pkgpath != NULL)
	  printf("%s", pkgpath);
	else if (prefix != NULL)
	  printf("%s.%s", prefix, package);
	else
	  printf("%s", package);
	printf(".%s\");\n", name);
	write_gcc_return_type(package, name, rets);
	printf(" %s_%s(", package, name);
	first = 1;
	write_params(params, &first);
	printf(")\n{\n");
	for (p = rets; p != NULL; p = p->next)
		printf("  %s %s;\n", p->type, p->name);
}

/* Write out a gcc function trailer.  */
static void
write_gcc_func_trailer(char *package, char *name, struct params *rets)
{
	if (rets == NULL)
		;
	else if (rets->next == NULL)
		printf("return %s;\n", rets->name);
	else {
		struct params *p;

		printf("  {\n    struct %s_%s_ret __ret;\n", package, name);
		for (p = rets; p != NULL; p = p->next)
			printf("    __ret.%s = %s;\n", p->name, p->name);
		printf("    return __ret;\n  }\n");
	}
	printf("}\n");
}

/* Write out a function header.  */
static void
write_func_header(char *package, char *name, struct params *params, 
		  struct params *rets)
{
	write_gcc_func_header(package, name, params, rets);
	printf("#line %d \"%s\"\n", lineno, file);
}

/* Write out a function trailer.  */
static void
write_func_trailer(char *package, char *name,
		   struct params *rets)
{
	write_gcc_func_trailer(package, name, rets);
}

/*
 * Read and write the body of the function, ending in an unnested }
 * (which is read but not written).
 */
static void
copy_body(void)
{
	int nesting = 0;
	while (1) {
		int c;

		c = getchar_no_eof();
		if (c == '}' && nesting == 0)
			return;
		putchar(c);
		switch (c) {
		default:
			break;
		case '{':
			++nesting;
			break;
		case '}':
			--nesting;
			break;
		case '/':
			c = getchar_update_lineno();
			putchar(c);
			if (c == '/') {
				do {
					c = getchar_no_eof();
					putchar(c);
				} while (c != '\n');
			} else if (c == '*') {
				while (1) {
					c = getchar_no_eof();
					putchar(c);
					if (c == '*') {
						do {
							c = getchar_no_eof();
							putchar(c);
						} while (c == '*');
						if (c == '/')
							break;
					}
				}
			}
			break;
		case '"':
		case '\'':
			{
				int delim = c;
				do {
					c = getchar_no_eof();
					putchar(c);
					if (c == '\\') {
						c = getchar_no_eof();
						putchar(c);
						c = '\0';
					}
				} while (c != delim);
			}
			break;
		}
	}
}

/* Process the entire file.  */
static void
process_file(void)
{
	char *package, *name;
	struct params *params, *rets;

	package = read_package();
	read_preprocessor_lines();
	while (read_func_header(&name, &params, &rets)) {
		char *p;
		char *pkg;
		char *nm;

		p = strchr(name, '.');
		if (p == NULL) {
			pkg = package;
			nm = name;
		} else {
			pkg = name;
			nm = p + 1;
			*p = '\0';
		}
		write_func_header(pkg, nm, params, rets);
		copy_body();
		write_func_trailer(pkg, nm, rets);
		free(name);
		free_params(params);
		free_params(rets);
	}
	free(package);
}

static void
usage(void)
{
	sysfatal("Usage: goc2c [--go-pkgpath PKGPATH] [--go-prefix PREFIX] [file]\n");
}

int
main(int argc, char **argv)
{
	char *goarch;

	argv0 = argv[0];
	while(argc > 1 && argv[1][0] == '-') {
		if(strcmp(argv[1], "-") == 0)
			break;
		if (strcmp(argv[1], "--go-pkgpath") == 0 && argc > 2) {
			pkgpath = argv[2];
			argc--;
			argv++;
		} else if (strcmp(argv[1], "--go-prefix") == 0 && argc > 2) {
			prefix = argv[2];
			argc--;
			argv++;
		} else
			usage();
		argc--;
		argv++;
	}

	if(argc <= 1 || strcmp(argv[1], "-") == 0) {
		file = "<stdin>";
		process_file();
		exit(0);
	}

	if(argc > 2)
		usage();

	file = argv[1];
	if(freopen(file, "r", stdin) == 0) {
		sysfatal("open %s: %r\n", file);
	}

	printf("// AUTO-GENERATED by autogen.sh; DO NOT EDIT\n\n");
	process_file();
	exit(0);
}
