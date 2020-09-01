/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France 
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 * and Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_id_private.h>
#include <isl/set.h>
#include <isl_seq.h>
#include <isl_stream_private.h>
#include <isl/obj.h>
#include "isl_polynomial_private.h"
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl_mat_private.h>
#include <isl_aff_private.h>
#include <isl_vec_private.h>
#include <isl/list.h>
#include <isl_val_private.h>

struct variable {
	char    	    	*name;
	int	     		 pos;
	struct variable		*next;
};

struct vars {
	struct isl_ctx	*ctx;
	int		 n;
	struct variable	*v;
};

static struct vars *vars_new(struct isl_ctx *ctx)
{
	struct vars *v;
	v = isl_alloc_type(ctx, struct vars);
	if (!v)
		return NULL;
	v->ctx = ctx;
	v->n = 0;
	v->v = NULL;
	return v;
}

static void variable_free(struct variable *var)
{
	while (var) {
		struct variable *next = var->next;
		free(var->name);
		free(var);
		var = next;
	}
}

static void vars_free(struct vars *v)
{
	if (!v)
		return;
	variable_free(v->v);
	free(v);
}

static void vars_drop(struct vars *v, int n)
{
	struct variable *var;

	if (!v || !v->v)
		return;

	v->n -= n;

	var = v->v;
	while (--n >= 0) {
		struct variable *next = var->next;
		free(var->name);
		free(var);
		var = next;
	}
	v->v = var;
}

static struct variable *variable_new(struct vars *v, const char *name, int len,
				int pos)
{
	struct variable *var;
	var = isl_calloc_type(v->ctx, struct variable);
	if (!var)
		goto error;
	var->name = strdup(name);
	var->name[len] = '\0';
	var->pos = pos;
	var->next = v->v;
	return var;
error:
	variable_free(v->v);
	return NULL;
}

static int vars_pos(struct vars *v, const char *s, int len)
{
	int pos;
	struct variable *q;

	if (len == -1)
		len = strlen(s);
	for (q = v->v; q; q = q->next) {
		if (strncmp(q->name, s, len) == 0 && q->name[len] == '\0')
			break;
	}
	if (q)
		pos = q->pos;
	else {
		pos = v->n;
		v->v = variable_new(v, s, len, v->n);
		if (!v->v)
			return -1;
		v->n++;
	}
	return pos;
}

static int vars_add_anon(struct vars *v)
{
	v->v = variable_new(v, "", 0, v->n);

	if (!v->v)
		return -1;
	v->n++;

	return 0;
}

/* Obtain next token, with some preprocessing.
 * In particular, evaluate expressions of the form x^y,
 * with x and y values.
 */
static struct isl_token *next_token(__isl_keep isl_stream *s)
{
	struct isl_token *tok, *tok2;

	tok = isl_stream_next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE)
		return tok;
	if (!isl_stream_eat_if_available(s, '^'))
		return tok;
	tok2 = isl_stream_next_token(s);
	if (!tok2 || tok2->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok2, "expecting constant value");
		goto error;
	}

	isl_int_pow_ui(tok->u.v, tok->u.v, isl_int_get_ui(tok2->u.v));

	isl_token_free(tok2);
	return tok;
error:
	isl_token_free(tok);
	isl_token_free(tok2);
	return NULL;
}

/* Read an isl_val from "s".
 *
 * The following token sequences are recognized
 *
 *	"infty"		->	infty
 *	"-" "infty"	->	-infty
 *	"NaN"		->	NaN
 *	n "/" d		->	n/d
 *	v		->	v
 *
 * where n, d and v are integer constants.
 */
__isl_give isl_val *isl_stream_read_val(__isl_keep isl_stream *s)
{
	struct isl_token *tok = NULL;
	struct isl_token *tok2 = NULL;
	isl_val *val;

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		goto error;
	}
	if (tok->type == ISL_TOKEN_INFTY) {
		isl_token_free(tok);
		return isl_val_infty(s->ctx);
	}
	if (tok->type == '-' &&
	    isl_stream_eat_if_available(s, ISL_TOKEN_INFTY)) {
		isl_token_free(tok);
		return isl_val_neginfty(s->ctx);
	}
	if (tok->type == ISL_TOKEN_NAN) {
		isl_token_free(tok);
		return isl_val_nan(s->ctx);
	}
	if (tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting value");
		goto error;
	}

	if (isl_stream_eat_if_available(s, '/')) {
		tok2 = next_token(s);
		if (!tok2) {
			isl_stream_error(s, NULL, "unexpected EOF");
			goto error;
		}
		if (tok2->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok2, "expecting value");
			goto error;
		}
		val = isl_val_rat_from_isl_int(s->ctx, tok->u.v, tok2->u.v);
		val = isl_val_normalize(val);
	} else {
		val = isl_val_int_from_isl_int(s->ctx, tok->u.v);
	}

	isl_token_free(tok);
	isl_token_free(tok2);
	return val;
error:
	isl_token_free(tok);
	isl_token_free(tok2);
	return NULL;
}

/* Read an isl_val from "str".
 */
__isl_give isl_val *isl_val_read_from_str(isl_ctx *ctx, const char *str)
{
	isl_val *val;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	val = isl_stream_read_val(s);
	isl_stream_free(s);
	return val;
}

/* Perform an integer division on *f and
 * an integer value read from the stream.
 */
static isl_stat int_div_by_cst(__isl_keep isl_stream *s, isl_int *f)
{
	struct isl_token *tok;

	tok = next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting constant value");
		goto error;
	}

	isl_int_fdiv_q(*f, *f, tok->u.v);

	isl_token_free(tok);

	return isl_stat_ok;
error:
	isl_token_free(tok);
	return isl_stat_error;
}

static isl_stat accept_cst_factor(__isl_keep isl_stream *s, isl_int *f)
{
	struct isl_token *tok;

	tok = next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting constant value");
		goto error;
	}

	isl_int_mul(*f, *f, tok->u.v);

	isl_token_free(tok);

	if (isl_stream_eat_if_available(s, '*'))
		return accept_cst_factor(s, f);

	return isl_stat_ok;
error:
	isl_token_free(tok);
	return isl_stat_error;
}

/* Given an affine expression aff, return an affine expression
 * for aff % d, with d the next token on the stream, which is
 * assumed to be a constant.
 *
 * We introduce an integer division q = [aff/d] and the result
 * is set to aff - d q.
 */
static __isl_give isl_pw_aff *affine_mod(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_pw_aff *aff)
{
	struct isl_token *tok;
	isl_pw_aff *q;

	tok = next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting constant value");
		goto error;
	}

	q = isl_pw_aff_copy(aff);
	q = isl_pw_aff_scale_down(q, tok->u.v);
	q = isl_pw_aff_floor(q);
	q = isl_pw_aff_scale(q, tok->u.v);

	aff = isl_pw_aff_sub(aff, q);

	isl_token_free(tok);
	return aff;
error:
	isl_pw_aff_free(aff);
	isl_token_free(tok);
	return NULL;
}

static __isl_give isl_pw_aff *accept_affine(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v);
static __isl_give isl_pw_aff_list *accept_affine_list(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v);

static __isl_give isl_pw_aff *accept_minmax(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v)
{
	struct isl_token *tok;
	isl_pw_aff_list *list = NULL;
	int min;

	tok = isl_stream_next_token(s);
	if (!tok)
		goto error;
	min = tok->type == ISL_TOKEN_MIN;
	isl_token_free(tok);

	if (isl_stream_eat(s, '('))
		goto error;

	list = accept_affine_list(s, isl_space_copy(space), v);
	if (!list)
		goto error;

	if (isl_stream_eat(s, ')'))
		goto error;

	isl_space_free(space);
	return min ? isl_pw_aff_list_min(list) : isl_pw_aff_list_max(list);
error:
	isl_space_free(space);
	isl_pw_aff_list_free(list);
	return NULL;
}

/* Is "tok" the start of an integer division?
 */
static int is_start_of_div(struct isl_token *tok)
{
	if (!tok)
		return 0;
	if (tok->type == '[')
		return 1;
	if (tok->type == ISL_TOKEN_FLOOR)
		return 1;
	if (tok->type == ISL_TOKEN_CEIL)
		return 1;
	if (tok->type == ISL_TOKEN_FLOORD)
		return 1;
	if (tok->type == ISL_TOKEN_CEILD)
		return 1;
	return 0;
}

/* Read an integer division from "s" and return it as an isl_pw_aff.
 *
 * The integer division can be of the form
 *
 *	[<affine expression>]
 *	floor(<affine expression>)
 *	ceil(<affine expression>)
 *	floord(<affine expression>,<denominator>)
 *	ceild(<affine expression>,<denominator>)
 */
static __isl_give isl_pw_aff *accept_div(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v)
{
	struct isl_token *tok;
	int f = 0;
	int c = 0;
	int extra = 0;
	isl_pw_aff *pwaff = NULL;

	if (isl_stream_eat_if_available(s, ISL_TOKEN_FLOORD))
		extra = f = 1;
	else if (isl_stream_eat_if_available(s, ISL_TOKEN_CEILD))
		extra = c = 1;
	else if (isl_stream_eat_if_available(s, ISL_TOKEN_FLOOR))
		f = 1;
	else if (isl_stream_eat_if_available(s, ISL_TOKEN_CEIL))
		c = 1;
	if (f || c) {
		if (isl_stream_eat(s, '('))
			goto error;
	} else {
		if (isl_stream_eat(s, '['))
			goto error;
	}

	pwaff = accept_affine(s, isl_space_copy(space), v);

	if (extra) {
		if (isl_stream_eat(s, ','))
			goto error;

		tok = next_token(s);
		if (!tok)
			goto error;
		if (tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok, "expected denominator");
			isl_stream_push_token(s, tok);
			goto error;
		}
		pwaff = isl_pw_aff_scale_down(pwaff,  tok->u.v);
		isl_token_free(tok);
	}

	if (c)
		pwaff = isl_pw_aff_ceil(pwaff);
	else
		pwaff = isl_pw_aff_floor(pwaff);

	if (f || c) {
		if (isl_stream_eat(s, ')'))
			goto error;
	} else {
		if (isl_stream_eat(s, ']'))
			goto error;
	}

	isl_space_free(space);
	return pwaff;
error:
	isl_space_free(space);
	isl_pw_aff_free(pwaff);
	return NULL;
}

/* Divide "pa" by an integer constant read from the stream.
 */
static __isl_give isl_pw_aff *pw_aff_div_by_cst(__isl_keep isl_stream *s,
	__isl_take isl_pw_aff *pa)
{
	isl_int f;
	isl_int_init(f);
	isl_int_set_si(f, 1);
	if (accept_cst_factor(s, &f) < 0)
		pa = isl_pw_aff_free(pa);
	pa = isl_pw_aff_scale_down(pa, f);
	isl_int_clear(f);

	return pa;
}

static __isl_give isl_pw_aff *accept_affine_factor(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v)
{
	struct isl_token *tok = NULL;
	isl_pw_aff *res = NULL;

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		goto error;
	}

	if (tok->type == ISL_TOKEN_AFF) {
		res = isl_pw_aff_copy(tok->u.pwaff);
		isl_token_free(tok);
	} else if (tok->type == ISL_TOKEN_IDENT) {
		int n = v->n;
		int pos = vars_pos(v, tok->u.s, -1);
		isl_aff *aff;

		if (pos < 0)
			goto error;
		if (pos >= n) {
			vars_drop(v, v->n - n);
			isl_stream_error(s, tok, "unknown identifier");
			goto error;
		}

		aff = isl_aff_zero_on_domain(isl_local_space_from_space(isl_space_copy(space)));
		if (!aff)
			goto error;
		isl_int_set_si(aff->v->el[2 + pos], 1);
		res = isl_pw_aff_from_aff(aff);
		isl_token_free(tok);
	} else if (tok->type == ISL_TOKEN_VALUE) {
		if (isl_stream_eat_if_available(s, '*')) {
			res = accept_affine_factor(s, isl_space_copy(space), v);
			res = isl_pw_aff_scale(res, tok->u.v);
		} else {
			isl_local_space *ls;
			isl_aff *aff;
			ls = isl_local_space_from_space(isl_space_copy(space));
			aff = isl_aff_zero_on_domain(ls);
			aff = isl_aff_add_constant(aff, tok->u.v);
			res = isl_pw_aff_from_aff(aff);
		}
		isl_token_free(tok);
	} else if (tok->type == '(') {
		isl_token_free(tok);
		tok = NULL;
		res = accept_affine(s, isl_space_copy(space), v);
		if (!res)
			goto error;
		if (isl_stream_eat(s, ')'))
			goto error;
	} else if (is_start_of_div(tok)) {
		isl_stream_push_token(s, tok);
		tok = NULL;
		res = accept_div(s, isl_space_copy(space), v);
	} else if (tok->type == ISL_TOKEN_MIN || tok->type == ISL_TOKEN_MAX) {
		isl_stream_push_token(s, tok);
		tok = NULL;
		res = accept_minmax(s, isl_space_copy(space), v);
	} else {
		isl_stream_error(s, tok, "expecting factor");
		goto error;
	}
	if (isl_stream_eat_if_available(s, '%') ||
	    isl_stream_eat_if_available(s, ISL_TOKEN_MOD)) {
		isl_space_free(space);
		return affine_mod(s, v, res);
	}
	if (isl_stream_eat_if_available(s, '*')) {
		isl_int f;
		isl_int_init(f);
		isl_int_set_si(f, 1);
		if (accept_cst_factor(s, &f) < 0) {
			isl_int_clear(f);
			goto error2;
		}
		res = isl_pw_aff_scale(res, f);
		isl_int_clear(f);
	}
	if (isl_stream_eat_if_available(s, '/'))
		res = pw_aff_div_by_cst(s, res);
	if (isl_stream_eat_if_available(s, ISL_TOKEN_INT_DIV))
		res = isl_pw_aff_floor(pw_aff_div_by_cst(s, res));

	isl_space_free(space);
	return res;
error:
	isl_token_free(tok);
error2:
	isl_pw_aff_free(res);
	isl_space_free(space);
	return NULL;
}

static __isl_give isl_pw_aff *add_cst(__isl_take isl_pw_aff *pwaff, isl_int v)
{
	isl_aff *aff;
	isl_space *space;

	space = isl_pw_aff_get_domain_space(pwaff);
	aff = isl_aff_zero_on_domain(isl_local_space_from_space(space));
	aff = isl_aff_add_constant(aff, v);

	return isl_pw_aff_add(pwaff, isl_pw_aff_from_aff(aff));
}

/* Return a piecewise affine expression defined on the specified domain
 * that represents NaN.
 */
static __isl_give isl_pw_aff *nan_on_domain(__isl_keep isl_space *space)
{
	isl_local_space *ls;

	ls = isl_local_space_from_space(isl_space_copy(space));
	return isl_pw_aff_nan_on_domain(ls);
}

static __isl_give isl_pw_aff *accept_affine(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v)
{
	struct isl_token *tok = NULL;
	isl_local_space *ls;
	isl_pw_aff *res;
	int sign = 1;

	ls = isl_local_space_from_space(isl_space_copy(space));
	res = isl_pw_aff_from_aff(isl_aff_zero_on_domain(ls));
	if (!res)
		goto error;

	for (;;) {
		tok = next_token(s);
		if (!tok) {
			isl_stream_error(s, NULL, "unexpected EOF");
			goto error;
		}
		if (tok->type == '-') {
			sign = -sign;
			isl_token_free(tok);
			continue;
		}
		if (tok->type == '(' || is_start_of_div(tok) ||
		    tok->type == ISL_TOKEN_MIN || tok->type == ISL_TOKEN_MAX ||
		    tok->type == ISL_TOKEN_IDENT ||
		    tok->type == ISL_TOKEN_AFF) {
			isl_pw_aff *term;
			isl_stream_push_token(s, tok);
			tok = NULL;
			term = accept_affine_factor(s,
						    isl_space_copy(space), v);
			if (sign < 0)
				res = isl_pw_aff_sub(res, term);
			else
				res = isl_pw_aff_add(res, term);
			if (!res)
				goto error;
			sign = 1;
		} else if (tok->type == ISL_TOKEN_VALUE) {
			if (sign < 0)
				isl_int_neg(tok->u.v, tok->u.v);
			if (isl_stream_eat_if_available(s, '*') ||
			    isl_stream_next_token_is(s, ISL_TOKEN_IDENT)) {
				isl_pw_aff *term;
				term = accept_affine_factor(s,
						    isl_space_copy(space), v);
				term = isl_pw_aff_scale(term, tok->u.v);
				res = isl_pw_aff_add(res, term);
				if (!res)
					goto error;
			} else {
				if (isl_stream_eat_if_available(s,
							ISL_TOKEN_INT_DIV) &&
				    int_div_by_cst(s, &tok->u.v) < 0)
					goto error;
				res = add_cst(res, tok->u.v);
			}
			sign = 1;
		} else if (tok->type == ISL_TOKEN_NAN) {
			res = isl_pw_aff_add(res, nan_on_domain(space));
		} else {
			isl_stream_error(s, tok, "unexpected isl_token");
			isl_stream_push_token(s, tok);
			isl_pw_aff_free(res);
			isl_space_free(space);
			return NULL;
		}
		isl_token_free(tok);

		tok = next_token(s);
		if (tok && tok->type == '-') {
			sign = -sign;
			isl_token_free(tok);
		} else if (tok && tok->type == '+') {
			/* nothing */
			isl_token_free(tok);
		} else if (tok && tok->type == ISL_TOKEN_VALUE &&
			   isl_int_is_neg(tok->u.v)) {
			isl_stream_push_token(s, tok);
		} else {
			if (tok)
				isl_stream_push_token(s, tok);
			break;
		}
	}

	isl_space_free(space);
	return res;
error:
	isl_space_free(space);
	isl_token_free(tok);
	isl_pw_aff_free(res);
	return NULL;
}

/* Is "type" the type of a comparison operator between lists
 * of affine expressions?
 */
static int is_list_comparator_type(int type)
{
	switch (type) {
	case ISL_TOKEN_LEX_LT:
	case ISL_TOKEN_LEX_GT:
	case ISL_TOKEN_LEX_LE:
	case ISL_TOKEN_LEX_GE:
		return 1;
	default:
		return 0;
	}
}

static int is_comparator(struct isl_token *tok)
{
	if (!tok)
		return 0;
	if (is_list_comparator_type(tok->type))
		return 1;

	switch (tok->type) {
	case ISL_TOKEN_LT:
	case ISL_TOKEN_GT:
	case ISL_TOKEN_LE:
	case ISL_TOKEN_GE:
	case ISL_TOKEN_NE:
	case '=':
		return 1;
	default:
		return 0;
	}
}

static __isl_give isl_map *read_formula(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational);
static __isl_give isl_pw_aff *accept_extended_affine(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v, int rational);

/* Accept a ternary operator, given the first argument.
 */
static __isl_give isl_pw_aff *accept_ternary(__isl_keep isl_stream *s,
	__isl_take isl_map *cond, struct vars *v, int rational)
{
	isl_space *space;
	isl_pw_aff *pwaff1 = NULL, *pwaff2 = NULL, *pa_cond;

	if (!cond)
		return NULL;

	if (isl_stream_eat(s, '?'))
		goto error;

	space = isl_space_wrap(isl_map_get_space(cond));
	pwaff1 = accept_extended_affine(s, space, v, rational);
	if (!pwaff1)
		goto error;

	if (isl_stream_eat(s, ':'))
		goto error;

	space = isl_pw_aff_get_domain_space(pwaff1);
	pwaff2 = accept_extended_affine(s, space, v, rational);
	if (!pwaff2)
		goto error;

	pa_cond = isl_set_indicator_function(isl_map_wrap(cond));
	return isl_pw_aff_cond(pa_cond, pwaff1, pwaff2);
error:
	isl_map_free(cond);
	isl_pw_aff_free(pwaff1);
	isl_pw_aff_free(pwaff2);
	return NULL;
}

/* Set *line and *col to those of the next token, if any.
 */
static void set_current_line_col(__isl_keep isl_stream *s, int *line, int *col)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return;

	*line = tok->line;
	*col = tok->col;
	isl_stream_push_token(s, tok);
}

/* Push a token encapsulating "pa" onto "s", with the given
 * line and column.
 */
static isl_stat push_aff(__isl_keep isl_stream *s, int line, int col,
	__isl_take isl_pw_aff *pa)
{
	struct isl_token *tok;

	tok = isl_token_new(s->ctx, line, col, 0);
	if (!tok)
		goto error;
	tok->type = ISL_TOKEN_AFF;
	tok->u.pwaff = pa;
	isl_stream_push_token(s, tok);

	return isl_stat_ok;
error:
	isl_pw_aff_free(pa);
	return isl_stat_error;
}

/* Is the next token a comparison operator?
 */
static int next_is_comparator(__isl_keep isl_stream *s)
{
	int is_comp;
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;

	is_comp = is_comparator(tok);
	isl_stream_push_token(s, tok);

	return is_comp;
}

/* Accept an affine expression that may involve ternary operators.
 * We first read an affine expression.
 * If it is not followed by a comparison operator, we simply return it.
 * Otherwise, we assume the affine expression is part of the first
 * argument of a ternary operator and try to parse that.
 */
static __isl_give isl_pw_aff *accept_extended_affine(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v, int rational)
{
	isl_map *cond;
	isl_pw_aff *pwaff;
	int line = -1, col = -1;

	set_current_line_col(s, &line, &col);

	pwaff = accept_affine(s, space, v);
	if (rational)
		pwaff = isl_pw_aff_set_rational(pwaff);
	if (!pwaff)
		return NULL;
	if (!next_is_comparator(s))
		return pwaff;

	space = isl_pw_aff_get_domain_space(pwaff);
	cond = isl_map_universe(isl_space_unwrap(space));

	if (push_aff(s, line, col, pwaff) < 0)
		cond = isl_map_free(cond);
	if (!cond)
		return NULL;

	cond = read_formula(s, v, cond, rational);

	return accept_ternary(s, cond, v, rational);
}

static __isl_give isl_map *read_var_def(__isl_keep isl_stream *s,
	__isl_take isl_map *map, enum isl_dim_type type, struct vars *v,
	int rational)
{
	isl_pw_aff *def;
	isl_size pos;
	isl_map *def_map;

	if (type == isl_dim_param)
		pos = isl_map_dim(map, isl_dim_param);
	else {
		pos = isl_map_dim(map, isl_dim_in);
		if (type == isl_dim_out) {
			isl_size n_out = isl_map_dim(map, isl_dim_out);
			if (pos < 0 || n_out < 0)
				return isl_map_free(map);
			pos += n_out;
		}
		type = isl_dim_in;
	}
	if (pos < 0)
		return isl_map_free(map);
	--pos;

	def = accept_extended_affine(s, isl_space_wrap(isl_map_get_space(map)),
					v, rational);
	def_map = isl_map_from_pw_aff(def);
	def_map = isl_map_equate(def_map, type, pos, isl_dim_out, 0);
	def_map = isl_set_unwrap(isl_map_domain(def_map));

	map = isl_map_intersect(map, def_map);

	return map;
}

static __isl_give isl_pw_aff_list *accept_affine_list(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v)
{
	isl_pw_aff *pwaff;
	isl_pw_aff_list *list;
	struct isl_token *tok = NULL;

	pwaff = accept_affine(s, isl_space_copy(space), v);
	list = isl_pw_aff_list_from_pw_aff(pwaff);
	if (!list)
		goto error;

	for (;;) {
		tok = isl_stream_next_token(s);
		if (!tok) {
			isl_stream_error(s, NULL, "unexpected EOF");
			goto error;
		}
		if (tok->type != ',') {
			isl_stream_push_token(s, tok);
			break;
		}
		isl_token_free(tok);

		pwaff = accept_affine(s, isl_space_copy(space), v);
		list = isl_pw_aff_list_concat(list,
				isl_pw_aff_list_from_pw_aff(pwaff));
		if (!list)
			goto error;
	}

	isl_space_free(space);
	return list;
error:
	isl_space_free(space);
	isl_pw_aff_list_free(list);
	return NULL;
}

static __isl_give isl_map *read_defined_var_list(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	struct isl_token *tok;

	while ((tok = isl_stream_next_token(s)) != NULL) {
		int p;
		int n = v->n;

		if (tok->type != ISL_TOKEN_IDENT)
			break;

		p = vars_pos(v, tok->u.s, -1);
		if (p < 0)
			goto error;
		if (p < n) {
			isl_stream_error(s, tok, "expecting unique identifier");
			goto error;
		}

		map = isl_map_add_dims(map, isl_dim_out, 1);

		isl_token_free(tok);
		tok = isl_stream_next_token(s);
		if (tok && tok->type == '=') {
			isl_token_free(tok);
			map = read_var_def(s, map, isl_dim_out, v, rational);
			tok = isl_stream_next_token(s);
		}

		if (!tok || tok->type != ',')
			break;

		isl_token_free(tok);
	}
	if (tok)
		isl_stream_push_token(s, tok);

	return map;
error:
	isl_token_free(tok);
	isl_map_free(map);
	return NULL;
}

static int next_is_tuple(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	int is_tuple;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type == '[') {
		isl_stream_push_token(s, tok);
		return 1;
	}
	if (tok->type != ISL_TOKEN_IDENT && !tok->is_keyword) {
		isl_stream_push_token(s, tok);
		return 0;
	}

	is_tuple = isl_stream_next_token_is(s, '[');

	isl_stream_push_token(s, tok);

	return is_tuple;
}

/* Does the next token mark the end of a tuple element?
 */
static int next_is_end_tuple_element(__isl_keep isl_stream *s)
{
	return isl_stream_next_token_is(s, ',') ||
	    isl_stream_next_token_is(s, ']');
}

/* Is the next token one that necessarily forms the start of a condition?
 */
static int next_is_condition_start(__isl_keep isl_stream *s)
{
	return isl_stream_next_token_is(s, ISL_TOKEN_EXISTS) ||
	    isl_stream_next_token_is(s, ISL_TOKEN_NOT) ||
	    isl_stream_next_token_is(s, ISL_TOKEN_TRUE) ||
	    isl_stream_next_token_is(s, ISL_TOKEN_FALSE) ||
	    isl_stream_next_token_is(s, ISL_TOKEN_MAP);
}

/* Is "pa" an expression in term of earlier dimensions?
 * The alternative is that the dimension is defined to be equal to itself,
 * meaning that it has a universe domain and an expression that depends
 * on itself.  "i" is the position of the expression in a sequence
 * of "n" expressions.  The final dimensions of "pa" correspond to
 * these "n" expressions.
 */
static isl_bool pw_aff_is_expr(__isl_keep isl_pw_aff *pa, int i, int n)
{
	isl_aff *aff;

	if (!pa)
		return isl_bool_error;
	if (pa->n != 1)
		return isl_bool_true;
	if (!isl_set_plain_is_universe(pa->p[0].set))
		return isl_bool_true;

	aff = pa->p[0].aff;
	if (isl_int_is_zero(aff->v->el[aff->v->size - n + i]))
		return isl_bool_true;
	return isl_bool_false;
}

/* Does the tuple contain any dimensions that are defined
 * in terms of earlier dimensions?
 */
static isl_bool tuple_has_expr(__isl_keep isl_multi_pw_aff *tuple)
{
	int i;
	isl_size n;
	isl_bool has_expr = isl_bool_false;
	isl_pw_aff *pa;

	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	if (n < 0)
		return isl_bool_error;
	for (i = 0; i < n; ++i) {
		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		has_expr = pw_aff_is_expr(pa, i, n);
		isl_pw_aff_free(pa);
		if (has_expr < 0 || has_expr)
			break;
	}

	return has_expr;
}

/* Set the name of dimension "pos" in "space" to "name".
 * During printing, we add primes if the same name appears more than once
 * to distinguish the occurrences.  Here, we remove those primes from "name"
 * before setting the name of the dimension.
 */
static __isl_give isl_space *space_set_dim_name(__isl_take isl_space *space,
	int pos, char *name)
{
	char *prime;

	if (!name)
		return space;

	prime = strchr(name, '\'');
	if (prime)
		*prime = '\0';
	space = isl_space_set_dim_name(space, isl_dim_out, pos, name);
	if (prime)
		*prime = '\'';

	return space;
}

/* Construct an isl_pw_aff defined on a "space" (with v->n variables)
 * that is equal to the last of those variables.
 */
static __isl_give isl_pw_aff *identity_tuple_el_on_space(
	__isl_take isl_space *space, struct vars *v)
{
	isl_aff *aff;

	aff = isl_aff_zero_on_domain(isl_local_space_from_space(space));
	aff = isl_aff_add_coefficient_si(aff, isl_dim_in, v->n - 1, 1);
	return isl_pw_aff_from_aff(aff);
}

/* Construct an isl_pw_aff defined on the domain space of "pa"
 * that is equal to the last variable in "v".
 *
 * That is, if D is the domain space of "pa", then construct
 *
 *	D[..., i] -> i.
 */
static __isl_give isl_pw_aff *init_range(__isl_keep isl_pw_aff *pa,
	struct vars *v)
{
	isl_space *space;

	space = isl_pw_aff_get_domain_space(pa);
	return identity_tuple_el_on_space(space, v);
}

/* Impose the lower bound "lower" on the variable represented by "range_pa".
 *
 * In particular, "range_pa" is of the form
 *
 *	D[..., i] -> i : C
 *
 * with D also the domains space of "lower' and "C" some constraints.
 *
 * Return the expression
 *
 *	D[..., i] -> i : C and i >= lower
 */
static __isl_give isl_pw_aff *set_lower(__isl_take isl_pw_aff *range_pa,
	__isl_take isl_pw_aff *lower)
{
	isl_set *range;

	range = isl_pw_aff_ge_set(isl_pw_aff_copy(range_pa), lower);
	return isl_pw_aff_intersect_domain(range_pa, range);
}

/* Impose the upper bound "upper" on the variable represented by "range_pa".
 *
 * In particular, "range_pa" is of the form
 *
 *	D[..., i] -> i : C
 *
 * with D also the domains space of "upper' and "C" some constraints.
 *
 * Return the expression
 *
 *	D[..., i] -> i : C and i <= upper
 */
static __isl_give isl_pw_aff *set_upper(__isl_take isl_pw_aff *range_pa,
	__isl_take isl_pw_aff *upper)
{
	isl_set *range;

	range = isl_pw_aff_le_set(isl_pw_aff_copy(range_pa), upper);
	return isl_pw_aff_intersect_domain(range_pa, range);
}

/* Construct a piecewise affine expression corresponding
 * to the last variable in "v" that is greater than or equal to "pa".
 *
 * In particular, if D is the domain space of "pa",
 * then construct the expression
 *
 *	D[..., i] -> i,
 *
 * impose lower bound "pa" and return
 *
 *	D[..., i] -> i : i >= pa
 */
static __isl_give isl_pw_aff *construct_lower(__isl_take isl_pw_aff *pa,
	struct vars *v)
{
	return set_lower(init_range(pa, v), pa);
}

/* Construct a piecewise affine expression corresponding
 * to the last variable in "v" that is smaller than or equal to "pa".
 *
 * In particular, if D is the domain space of "pa",
 * then construct the expression
 *
 *	D[..., i] -> i,
 *
 * impose lower bound "pa" and return
 *
 *	D[..., i] -> i : i <= pa
 */
static __isl_give isl_pw_aff *construct_upper(__isl_take isl_pw_aff *pa,
	struct vars *v)
{
	return set_upper(init_range(pa, v), pa);
}

/* Construct a piecewise affine expression corresponding
 * to the last variable in "v" that ranges between "pa" and "pa2".
 *
 * In particular, if D is the domain space of "pa" (and "pa2"),
 * then construct the expression
 *
 *	D[..., i] -> i,
 *
 * impose lower bound "pa" and upper bound "pa2" and return
 *
 *	D[..., i] -> i : pa <= i <= pa2
 */
static __isl_give isl_pw_aff *construct_range(__isl_take isl_pw_aff *pa,
	__isl_take isl_pw_aff *pa2, struct vars *v)
{
	return set_upper(set_lower(init_range(pa, v), pa), pa2);
}

static int resolve_paren_expr(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational);

/* Given that the (piecewise) affine expression "pa"
 * has just been parsed, followed by a colon,
 * continue parsing as part of a piecewise affine expression.
 *
 * In particular, check if the colon is followed by a condition.
 * If so, parse the conditions(a) on "pa" and include them in the domain.
 * Otherwise, if the colon is followed by another (piecewise) affine expression
 * then consider the two expressions as endpoints of a range of values and
 * return a piecewise affine expression that takes values in that range.
 * Note that an affine expression followed by a comparison operator
 * is considered to be part of a condition.
 * If the colon is not followed by anything (inside the tuple element),
 * then consider "pa" as a lower bound on a range of values without upper bound
 * and return a piecewise affine expression that takes values in that range.
 */
static __isl_give isl_pw_aff *update_piecewise_affine_colon(
	__isl_take isl_pw_aff *pa, __isl_keep isl_stream *s,
	struct vars *v, int rational)
{
	isl_space *dom_space;
	isl_map *map;

	dom_space = isl_pw_aff_get_domain_space(pa);
	map = isl_map_universe(isl_space_from_domain(dom_space));

	if (isl_stream_next_token_is(s, '('))
		if (resolve_paren_expr(s, v, isl_map_copy(map), rational))
			goto error;
	if (next_is_end_tuple_element(s)) {
		isl_map_free(map);
		return construct_lower(pa, v);
	}
	if (!next_is_condition_start(s)) {
		int line = -1, col = -1;
		isl_space *space;
		isl_pw_aff *pa2;

		set_current_line_col(s, &line, &col);
		space = isl_space_wrap(isl_map_get_space(map));
		pa2 = accept_affine(s, space, v);
		if (rational)
			pa2 = isl_pw_aff_set_rational(pa2);
		if (!next_is_comparator(s)) {
			isl_map_free(map);
			pa2 = isl_pw_aff_domain_factor_domain(pa2);
			return construct_range(pa, pa2, v);
		}
		if (push_aff(s, line, col, pa2) < 0)
			goto error;
	}

	map = read_formula(s, v, map, rational);
	pa = isl_pw_aff_intersect_domain(pa, isl_map_domain(map));

	return pa;
error:
	isl_map_free(map);
	isl_pw_aff_free(pa);
	return NULL;
}

/* Accept a piecewise affine expression.
 *
 * At the outer level, the piecewise affine expression may be of the form
 *
 *	aff1 : condition1; aff2 : conditions2; ...
 *
 * or one of
 *
 *	aff :
 *	aff1 : aff2
 *	: aff
 *	:
 *
 * or simply
 *
 *	aff
 *
 * each of the affine expressions may in turn include ternary operators.
 *
 * If the first token is a colon, then the expression must be
 * ":" or ": aff2", depending on whether anything follows the colon
 * inside the tuple element.
 * The first is considered to represent an arbitrary value.
 * The second is considered to represent a range of values
 * with the given upper bound and no lower bound.
 *
 * There may be parentheses around some subexpression of "aff1"
 * around "aff1" itself, around "aff1 : condition1" and/or
 * around the entire piecewise affine expression.
 * We therefore remove the opening parenthesis (if any) from the stream
 * in case the closing parenthesis follows the colon, but if the closing
 * parenthesis is the first thing in the stream after the parsed affine
 * expression, we push the parsed expression onto the stream and parse
 * again in case the parentheses enclose some subexpression of "aff1".
 */
static __isl_give isl_pw_aff *accept_piecewise_affine(__isl_keep isl_stream *s,
	__isl_take isl_space *space, struct vars *v, int rational)
{
	isl_pw_aff *res;
	isl_space *res_space;

	if (isl_stream_eat_if_available(s, ':')) {
		if (next_is_end_tuple_element(s))
			return identity_tuple_el_on_space(space, v);
		else
			return construct_upper(accept_affine(s, space, v), v);
	}

	res_space = isl_space_from_domain(isl_space_copy(space));
	res_space = isl_space_add_dims(res_space, isl_dim_out, 1);
	res = isl_pw_aff_empty(res_space);
	do {
		isl_pw_aff *pa;
		int seen_paren;
		int line = -1, col = -1;

		set_current_line_col(s, &line, &col);
		seen_paren = isl_stream_eat_if_available(s, '(');
		if (seen_paren)
			pa = accept_piecewise_affine(s, isl_space_copy(space),
							v, rational);
		else
			pa = accept_extended_affine(s, isl_space_copy(space),
							v, rational);
		if (seen_paren && isl_stream_eat_if_available(s, ')')) {
			seen_paren = 0;
			if (push_aff(s, line, col, pa) < 0)
				goto error;
			pa = accept_extended_affine(s, isl_space_copy(space),
							v, rational);
		}
		if (isl_stream_eat_if_available(s, ':'))
			pa = update_piecewise_affine_colon(pa, s, v, rational);

		res = isl_pw_aff_union_add(res, pa);

		if (seen_paren && isl_stream_eat(s, ')'))
			goto error;
	} while (isl_stream_eat_if_available(s, ';'));

	isl_space_free(space);

	return res;
error:
	isl_space_free(space);
	return isl_pw_aff_free(res);
}

/* Read an affine expression from "s" for use in read_tuple.
 *
 * accept_extended_affine requires a wrapped space as input.
 * read_tuple on the other hand expects each isl_pw_aff
 * to have an anonymous space.  We therefore adjust the space
 * of the isl_pw_aff before returning it.
 */
static __isl_give isl_pw_aff *read_tuple_var_def(__isl_keep isl_stream *s,
	struct vars *v, int rational)
{
	isl_space *space;
	isl_pw_aff *def;

	space = isl_space_wrap(isl_space_alloc(s->ctx, 0, v->n, 0));

	def = accept_piecewise_affine(s, space, v, rational);
	def = isl_pw_aff_domain_factor_domain(def);

	return def;
}

/* Read a list of tuple elements by calling "read_el" on each of them and
 * return a space with the same number of set dimensions derived from
 * the parameter space "space" and possibly updated by "read_el".
 * The elements in the list are separated by either "," or "][".
 * If "comma" is set then only "," is allowed.
 */
static __isl_give isl_space *read_tuple_list(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_space *space, int rational, int comma,
	__isl_give isl_space *(*read_el)(__isl_keep isl_stream *s,
		struct vars *v, __isl_take isl_space *space, int rational,
		void *user),
	void *user)
{
	if (!space)
		return NULL;

	space = isl_space_set_from_params(space);

	if (isl_stream_next_token_is(s, ']'))
		return space;

	for (;;) {
		struct isl_token *tok;

		space = isl_space_add_dims(space, isl_dim_set, 1);

		space = read_el(s, v, space, rational, user);
		if (!space)
			return NULL;

		tok = isl_stream_next_token(s);
		if (!comma && tok && tok->type == ']' &&
		    isl_stream_next_token_is(s, '[')) {
			isl_token_free(tok);
			tok = isl_stream_next_token(s);
		} else if (!tok || tok->type != ',') {
			if (tok)
				isl_stream_push_token(s, tok);
			break;
		}

		isl_token_free(tok);
	}

	return space;
}

/* Read a tuple space from "s" derived from the parameter space "space".
 * Call "read_el" on each element in the tuples.
 */
static __isl_give isl_space *read_tuple_space(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_space *space, int rational, int comma,
	__isl_give isl_space *(*read_el)(__isl_keep isl_stream *s,
		struct vars *v, __isl_take isl_space *space, int rational,
		void *user),
	void *user)
{
	struct isl_token *tok;
	char *name = NULL;
	isl_space *res = NULL;

	tok = isl_stream_next_token(s);
	if (!tok)
		goto error;
	if (tok->type == ISL_TOKEN_IDENT || tok->is_keyword) {
		name = strdup(tok->u.s);
		isl_token_free(tok);
		if (!name)
			goto error;
	} else
		isl_stream_push_token(s, tok);
	if (isl_stream_eat(s, '['))
		goto error;
	if (next_is_tuple(s)) {
		isl_space *out;
		res = read_tuple_space(s, v, isl_space_copy(space),
					rational, comma, read_el, user);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
		out = read_tuple_space(s, v, isl_space_copy(space),
					rational, comma, read_el, user);
		res = isl_space_product(res, out);
	} else
		res = read_tuple_list(s, v, isl_space_copy(space),
					rational, comma, read_el, user);
	if (isl_stream_eat(s, ']'))
		goto error;

	if (name) {
		res = isl_space_set_tuple_name(res, isl_dim_set, name);
		free(name);
	}

	isl_space_free(space);
	return res;
error:
	free(name);
	isl_space_free(res);
	isl_space_free(space);
	return NULL;
}

/* Construct an isl_pw_aff defined on a space with v->n variables
 * that is equal to the last of those variables.
 */
static __isl_give isl_pw_aff *identity_tuple_el(struct vars *v)
{
	isl_space *space;

	space = isl_space_set_alloc(v->ctx, 0, v->n);
	return identity_tuple_el_on_space(space, v);
}

/* This function is called for each element in a tuple inside read_tuple.
 * Add a new variable to "v" and construct a corresponding isl_pw_aff defined
 * over a space containing all variables in "v" defined so far.
 * The isl_pw_aff expresses the new variable in terms of earlier variables
 * if a definition is provided.  Otherwise, it is represented as being
 * equal to itself.
 * Add the isl_pw_aff to *list.
 * If the new variable was named, then adjust "space" accordingly and
 * return the updated space.
 */
static __isl_give isl_space *read_tuple_pw_aff_el(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_space *space, int rational, void *user)
{
	isl_pw_aff_list **list = (isl_pw_aff_list **) user;
	isl_pw_aff *pa;
	struct isl_token *tok;
	int new_name = 0;

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return isl_space_free(space);
	}

	if (tok->type == ISL_TOKEN_IDENT) {
		int n = v->n;
		int p = vars_pos(v, tok->u.s, -1);
		if (p < 0)
			goto error;
		new_name = p >= n;
	}

	if (tok->type == '*') {
		if (vars_add_anon(v) < 0)
			goto error;
		isl_token_free(tok);
		pa = identity_tuple_el(v);
	} else if (new_name) {
		isl_size pos = isl_space_dim(space, isl_dim_out);
		if (pos < 0)
			goto error;
		pos -= 1;
		space = space_set_dim_name(space, pos, v->v->name);
		isl_token_free(tok);
		if (isl_stream_eat_if_available(s, '='))
			pa = read_tuple_var_def(s, v, rational);
		else
			pa = identity_tuple_el(v);
	} else {
		isl_stream_push_token(s, tok);
		tok = NULL;
		if (vars_add_anon(v) < 0)
			goto error;
		pa = read_tuple_var_def(s, v, rational);
	}

	*list = isl_pw_aff_list_add(*list, pa);
	if (!*list)
		return isl_space_free(space);

	return space;
error:
	isl_token_free(tok);
	return isl_space_free(space);
}

/* Read a tuple and represent it as an isl_multi_pw_aff.
 * The range space of the isl_multi_pw_aff is the space of the tuple.
 * The domain space is an anonymous space
 * with a dimension for each variable in the set of variables in "v",
 * including the variables in the range.
 * If a given dimension is not defined in terms of earlier dimensions in
 * the input, then the corresponding isl_pw_aff is set equal to one time
 * the variable corresponding to the dimension being defined.
 *
 * The elements in the tuple are collected in a list by read_tuple_pw_aff_el.
 * Each element in this list is defined over a space representing
 * the variables defined so far.  We need to adjust the earlier
 * elements to have as many variables in the domain as the final
 * element in the list.
 */
static __isl_give isl_multi_pw_aff *read_tuple(__isl_keep isl_stream *s,
	struct vars *v, int rational, int comma)
{
	int i;
	isl_size n;
	isl_space *space;
	isl_pw_aff_list *list;

	space = isl_space_params_alloc(v->ctx, 0);
	list = isl_pw_aff_list_alloc(s->ctx, 0);
	space = read_tuple_space(s, v, space, rational, comma,
				&read_tuple_pw_aff_el, &list);
	n = isl_space_dim(space, isl_dim_set);
	if (n < 0)
		space = isl_space_free(space);
	for (i = 0; i + 1 < n; ++i) {
		isl_pw_aff *pa;

		pa = isl_pw_aff_list_get_pw_aff(list, i);
		pa = isl_pw_aff_add_dims(pa, isl_dim_in, n - (i + 1));
		list = isl_pw_aff_list_set_pw_aff(list, i, pa);
	}

	space = isl_space_from_range(space);
	space = isl_space_add_dims(space, isl_dim_in, v->n);
	return isl_multi_pw_aff_from_pw_aff_list(space, list);
}

/* Add the tuple represented by the isl_multi_pw_aff "tuple" to "map".
 * We first create the appropriate space in "map" based on the range
 * space of this isl_multi_pw_aff.  Then, we add equalities based
 * on the affine expressions.  These live in an anonymous space,
 * however, so we first need to reset the space to that of "map".
 */
static __isl_give isl_map *map_from_tuple(__isl_take isl_multi_pw_aff *tuple,
	__isl_take isl_map *map, enum isl_dim_type type, struct vars *v,
	int rational)
{
	int i;
	isl_size n;
	isl_ctx *ctx;
	isl_space *space = NULL;

	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	if (!map || n < 0)
		goto error;
	ctx = isl_multi_pw_aff_get_ctx(tuple);
	space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
	if (!space)
		goto error;

	if (type == isl_dim_param) {
		if (isl_space_has_tuple_name(space, isl_dim_set) ||
		    isl_space_is_wrapping(space)) {
			isl_die(ctx, isl_error_invalid,
				"parameter tuples cannot be named or nested",
				goto error);
		}
		map = isl_map_add_dims(map, type, n);
		for (i = 0; i < n; ++i) {
			isl_id *id;
			if (!isl_space_has_dim_name(space, isl_dim_set, i))
				isl_die(ctx, isl_error_invalid,
					"parameters must be named",
					goto error);
			id = isl_space_get_dim_id(space, isl_dim_set, i);
			map = isl_map_set_dim_id(map, isl_dim_param, i, id);
		}
	} else if (type == isl_dim_in) {
		isl_set *set;

		set = isl_set_universe(isl_space_copy(space));
		if (rational)
			set = isl_set_set_rational(set);
		set = isl_set_intersect_params(set, isl_map_params(map));
		map = isl_map_from_domain(set);
	} else {
		isl_set *set;

		set = isl_set_universe(isl_space_copy(space));
		if (rational)
			set = isl_set_set_rational(set);
		map = isl_map_from_domain_and_range(isl_map_domain(map), set);
	}

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		isl_space *space;
		isl_aff *aff;
		isl_set *set;
		isl_map *map_i;

		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		space = isl_pw_aff_get_domain_space(pa);
		aff = isl_aff_zero_on_domain(isl_local_space_from_space(space));
		aff = isl_aff_add_coefficient_si(aff,
						isl_dim_in, v->n - n + i, -1);
		pa = isl_pw_aff_add(pa, isl_pw_aff_from_aff(aff));
		if (rational)
			pa = isl_pw_aff_set_rational(pa);
		set = isl_pw_aff_zero_set(pa);
		map_i = isl_map_from_range(set);
		map_i = isl_map_reset_space(map_i, isl_map_get_space(map));
		map = isl_map_intersect(map, map_i);
	}

	isl_space_free(space);
	isl_multi_pw_aff_free(tuple);
	return map;
error:
	isl_space_free(space);
	isl_multi_pw_aff_free(tuple);
	isl_map_free(map);
	return NULL;
}

/* Read a tuple from "s" and add it to "map".
 * The tuple is initially represented as an isl_multi_pw_aff and
 * then added to "map".
 */
static __isl_give isl_map *read_map_tuple(__isl_keep isl_stream *s,
	__isl_take isl_map *map, enum isl_dim_type type, struct vars *v,
	int rational, int comma)
{
	isl_multi_pw_aff *tuple;

	tuple = read_tuple(s, v, rational, comma);
	if (!tuple)
		return isl_map_free(map);

	return map_from_tuple(tuple, map, type, v, rational);
}

/* Given two equal-length lists of piecewise affine expression with the space
 * of "set" as domain, construct a set in the same space that expresses
 * that "left" and "right" satisfy the comparison "type".
 *
 * A space is constructed of the same dimension as the number of elements
 * in the two lists.  The comparison is then expressed in a map from
 * this space to itself and wrapped into a set.  Finally the two lists
 * of piecewise affine expressions are plugged into this set.
 *
 * Let S be the space of "set" and T the constructed space.
 * The lists are first changed into two isl_multi_pw_affs in S -> T and
 * then combined into an isl_multi_pw_aff in S -> [T -> T],
 * while the comparison is first expressed in T -> T, then [T -> T]
 * and finally in S.
 */
static __isl_give isl_set *list_cmp(__isl_keep isl_set *set, int type,
	__isl_take isl_pw_aff_list *left, __isl_take isl_pw_aff_list *right)
{
	isl_space *space;
	isl_size n;
	isl_multi_pw_aff *mpa1, *mpa2;

	n = isl_pw_aff_list_n_pw_aff(left);
	if (!set || n < 0 || !right)
		goto error;

	space = isl_set_get_space(set);
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, n);
	mpa1 = isl_multi_pw_aff_from_pw_aff_list(isl_space_copy(space), left);
	mpa2 = isl_multi_pw_aff_from_pw_aff_list(isl_space_copy(space), right);
	mpa1 = isl_multi_pw_aff_range_product(mpa1, mpa2);

	space = isl_space_range(space);
	switch (type) {
	case ISL_TOKEN_LEX_LT:
		set = isl_map_wrap(isl_map_lex_lt(space));
		break;
	case ISL_TOKEN_LEX_GT:
		set = isl_map_wrap(isl_map_lex_gt(space));
		break;
	case ISL_TOKEN_LEX_LE:
		set = isl_map_wrap(isl_map_lex_le(space));
		break;
	case ISL_TOKEN_LEX_GE:
		set = isl_map_wrap(isl_map_lex_ge(space));
		break;
	default:
		isl_multi_pw_aff_free(mpa1);
		isl_space_free(space);
		isl_die(isl_set_get_ctx(set), isl_error_internal,
			"unhandled list comparison type", return NULL);
	}
	set = isl_set_preimage_multi_pw_aff(set, mpa1);
	return set;
error:
	isl_pw_aff_list_free(left);
	isl_pw_aff_list_free(right);
	return NULL;
}

/* Construct constraints of the form
 *
 *	a op b
 *
 * where a is an element in "left", op is an operator of type "type" and
 * b is an element in "right", add the constraints to "set" and return
 * the result.
 * "rational" is set if the constraints should be treated as
 * a rational constraints.
 *
 * If "type" is the type of a comparison operator between lists
 * of affine expressions, then a single (compound) constraint
 * is constructed by list_cmp instead.
 */
static __isl_give isl_set *construct_constraints(
	__isl_take isl_set *set, int type,
	__isl_keep isl_pw_aff_list *left, __isl_keep isl_pw_aff_list *right,
	int rational)
{
	isl_set *cond;

	left = isl_pw_aff_list_copy(left);
	right = isl_pw_aff_list_copy(right);
	if (rational) {
		left = isl_pw_aff_list_set_rational(left);
		right = isl_pw_aff_list_set_rational(right);
	}
	if (is_list_comparator_type(type))
		cond = list_cmp(set, type, left, right);
	else if (type == ISL_TOKEN_LE)
		cond = isl_pw_aff_list_le_set(left, right);
	else if (type == ISL_TOKEN_GE)
		cond = isl_pw_aff_list_ge_set(left, right);
	else if (type == ISL_TOKEN_LT)
		cond = isl_pw_aff_list_lt_set(left, right);
	else if (type == ISL_TOKEN_GT)
		cond = isl_pw_aff_list_gt_set(left, right);
	else if (type == ISL_TOKEN_NE)
		cond = isl_pw_aff_list_ne_set(left, right);
	else
		cond = isl_pw_aff_list_eq_set(left, right);

	return isl_set_intersect(set, cond);
}

/* Read a constraint from "s", add it to "map" and return the result.
 * "v" contains a description of the identifiers parsed so far.
 * "rational" is set if the constraint should be treated as
 * a rational constraint.
 * The constraint read from "s" may be applied to multiple pairs
 * of affine expressions and may be chained.
 * In particular, a list of affine expressions is read, followed
 * by a comparison operator and another list of affine expressions.
 * The comparison operator is then applied to each pair of elements
 * in the two lists and the results are added to "map".
 * However, if the operator expects two lists of affine expressions,
 * then it is applied directly to those lists and the two lists
 * are required to have the same length.
 * If the next token is another comparison operator, then another
 * list of affine expressions is read and the process repeats.
 *
 * The processing is performed on a wrapped copy of "map" because
 * an affine expression cannot have a binary relation as domain.
 */
static __isl_give isl_map *add_constraint(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	struct isl_token *tok;
	int type;
	isl_pw_aff_list *list1 = NULL, *list2 = NULL;
	isl_size n1, n2;
	isl_set *set;

	set = isl_map_wrap(map);
	list1 = accept_affine_list(s, isl_set_get_space(set), v);
	if (!list1)
		goto error;
	tok = isl_stream_next_token(s);
	if (!is_comparator(tok)) {
		isl_stream_error(s, tok, "missing operator");
		if (tok)
			isl_stream_push_token(s, tok);
		goto error;
	}
	type = tok->type;
	isl_token_free(tok);
	for (;;) {
		list2 = accept_affine_list(s, isl_set_get_space(set), v);
		n1 = isl_pw_aff_list_n_pw_aff(list1);
		n2 = isl_pw_aff_list_n_pw_aff(list2);
		if (n1 < 0 || n2 < 0)
			goto error;
		if (is_list_comparator_type(type) && n1 != n2) {
			isl_stream_error(s, NULL,
					"list arguments not of same size");
			goto error;
		}

		set = construct_constraints(set, type, list1, list2, rational);
		isl_pw_aff_list_free(list1);
		list1 = list2;

		if (!next_is_comparator(s))
			break;
		tok = isl_stream_next_token(s);
		type = tok->type;
		isl_token_free(tok);
	}
	isl_pw_aff_list_free(list1);

	return isl_set_unwrap(set);
error:
	isl_pw_aff_list_free(list1);
	isl_pw_aff_list_free(list2);
	isl_set_free(set);
	return NULL;
}

static __isl_give isl_map *read_exists(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	int n = v->n;
	int seen_paren = isl_stream_eat_if_available(s, '(');

	map = isl_map_from_domain(isl_map_wrap(map));
	map = read_defined_var_list(s, v, map, rational);

	if (isl_stream_eat(s, ':'))
		goto error;

	map = read_formula(s, v, map, rational);
	map = isl_set_unwrap(isl_map_domain(map));

	vars_drop(v, v->n - n);
	if (seen_paren && isl_stream_eat(s, ')'))
		goto error;

	return map;
error:
	isl_map_free(map);
	return NULL;
}

/* Parse an expression between parentheses and push the result
 * back on the stream.
 *
 * The parsed expression may be either an affine expression
 * or a condition.  The first type is pushed onto the stream
 * as an isl_pw_aff, while the second is pushed as an isl_map.
 *
 * If the initial token indicates the start of a condition,
 * we parse it as such.
 * Otherwise, we first parse an affine expression and push
 * that onto the stream.  If the affine expression covers the
 * entire expression between parentheses, we return.
 * Otherwise, we assume that the affine expression is the
 * start of a condition and continue parsing.
 */
static int resolve_paren_expr(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	struct isl_token *tok, *tok2;
	int has_paren;
	int line, col;
	isl_pw_aff *pwaff;

	tok = isl_stream_next_token(s);
	if (!tok || tok->type != '(')
		goto error;

	if (isl_stream_next_token_is(s, '('))
		if (resolve_paren_expr(s, v, isl_map_copy(map), rational))
			goto error;

	if (next_is_condition_start(s)) {
		map = read_formula(s, v, map, rational);
		if (isl_stream_eat(s, ')'))
			goto error;
		tok->type = ISL_TOKEN_MAP;
		tok->u.map = map;
		isl_stream_push_token(s, tok);
		return 0;
	}

	tok2 = isl_stream_next_token(s);
	if (!tok2)
		goto error;
	line = tok2->line;
	col = tok2->col;
	isl_stream_push_token(s, tok2);

	pwaff = accept_affine(s, isl_space_wrap(isl_map_get_space(map)), v);
	if (!pwaff)
		goto error;

	has_paren = isl_stream_eat_if_available(s, ')');

	if (push_aff(s, line, col, pwaff) < 0)
		goto error;

	if (has_paren) {
		isl_token_free(tok);
		isl_map_free(map);
		return 0;
	}

	map = read_formula(s, v, map, rational);
	if (isl_stream_eat(s, ')'))
		goto error;

	tok->type = ISL_TOKEN_MAP;
	tok->u.map = map;
	isl_stream_push_token(s, tok);

	return 0;
error:
	isl_token_free(tok);
	isl_map_free(map);
	return -1;
}

static __isl_give isl_map *read_conjunct(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	if (isl_stream_next_token_is(s, '('))
		if (resolve_paren_expr(s, v, isl_map_copy(map), rational))
			goto error;

	if (isl_stream_next_token_is(s, ISL_TOKEN_MAP)) {
		struct isl_token *tok;
		tok = isl_stream_next_token(s);
		if (!tok)
			goto error;
		isl_map_free(map);
		map = isl_map_copy(tok->u.map);
		isl_token_free(tok);
		return map;
	}

	if (isl_stream_eat_if_available(s, ISL_TOKEN_EXISTS))
		return read_exists(s, v, map, rational);

	if (isl_stream_eat_if_available(s, ISL_TOKEN_TRUE))
		return map;

	if (isl_stream_eat_if_available(s, ISL_TOKEN_FALSE)) {
		isl_space *space = isl_map_get_space(map);
		isl_map_free(map);
		return isl_map_empty(space);
	}
		
	return add_constraint(s, v, map, rational);
error:
	isl_map_free(map);
	return NULL;
}

static __isl_give isl_map *read_conjuncts(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	isl_map *res;
	int negate;

	negate = isl_stream_eat_if_available(s, ISL_TOKEN_NOT);
	res = read_conjunct(s, v, isl_map_copy(map), rational);
	if (negate)
		res = isl_map_subtract(isl_map_copy(map), res);

	while (res && isl_stream_eat_if_available(s, ISL_TOKEN_AND)) {
		isl_map *res_i;

		negate = isl_stream_eat_if_available(s, ISL_TOKEN_NOT);
		res_i = read_conjunct(s, v, isl_map_copy(map), rational);
		if (negate)
			res = isl_map_subtract(res, res_i);
		else
			res = isl_map_intersect(res, res_i);
	}

	isl_map_free(map);
	return res;
}

static __isl_give isl_map *read_disjuncts(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	isl_map *res;

	if (isl_stream_next_token_is(s, '}'))
		return map;

	res = read_conjuncts(s, v, isl_map_copy(map), rational);
	while (isl_stream_eat_if_available(s, ISL_TOKEN_OR)) {
		isl_map *res_i;

		res_i = read_conjuncts(s, v, isl_map_copy(map), rational);
		res = isl_map_union(res, res_i);
	}

	isl_map_free(map);
	return res;
}

/* Read a first order formula from "s", add the corresponding
 * constraints to "map" and return the result.
 *
 * In particular, read a formula of the form
 *
 *	a
 *
 * or
 *
 *	a implies b
 *
 * where a and b are disjunctions.
 *
 * In the first case, map is replaced by
 *
 *	map \cap { [..] : a }
 *
 * In the second case, it is replaced by
 *
 *	(map \setminus { [..] : a}) \cup (map \cap { [..] : b })
 */
static __isl_give isl_map *read_formula(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_map *map, int rational)
{
	isl_map *res;

	res = read_disjuncts(s, v, isl_map_copy(map), rational);

	if (isl_stream_eat_if_available(s, ISL_TOKEN_IMPLIES)) {
		isl_map *res2;

		res = isl_map_subtract(isl_map_copy(map), res);
		res2 = read_disjuncts(s, v, map, rational);
		res = isl_map_union(res, res2);
	} else
		isl_map_free(map);

	return res;
}

static isl_size polylib_pos_to_isl_pos(__isl_keep isl_basic_map *bmap, int pos)
{
	isl_size n_out, n_in, n_param, n_div;

	n_param = isl_basic_map_dim(bmap, isl_dim_param);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	n_div = isl_basic_map_dim(bmap, isl_dim_div);
	if (n_param < 0 || n_in < 0 || n_out < 0 || n_div < 0)
		return isl_size_error;

	if (pos < n_out)
		return 1 + n_param + n_in + pos;
	pos -= n_out;

	if (pos < n_in)
		return 1 + n_param + pos;
	pos -= n_in;

	if (pos < n_div)
		return 1 + n_param + n_in + n_out + pos;
	pos -= n_div;

	if (pos < n_param)
		return 1 + pos;

	return 0;
}

static __isl_give isl_basic_map *basic_map_read_polylib_constraint(
	__isl_keep isl_stream *s, __isl_take isl_basic_map *bmap)
{
	int j;
	struct isl_token *tok;
	int type;
	int k;
	isl_int *c;
	isl_size total;

	if (!bmap)
		return NULL;

	tok = isl_stream_next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting coefficient");
		if (tok)
			isl_stream_push_token(s, tok);
		goto error;
	}
	if (!tok->on_new_line) {
		isl_stream_error(s, tok, "coefficient should appear on new line");
		isl_stream_push_token(s, tok);
		goto error;
	}

	type = isl_int_get_si(tok->u.v);
	isl_token_free(tok);

	isl_assert(s->ctx, type == 0 || type == 1, goto error);
	if (type == 0) {
		k = isl_basic_map_alloc_equality(bmap);
		c = bmap->eq[k];
	} else {
		k = isl_basic_map_alloc_inequality(bmap);
		c = bmap->ineq[k];
	}
	if (k < 0)
		goto error;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)
		return isl_basic_map_free(bmap);
	for (j = 0; j < 1 + total; ++j) {
		isl_size pos;
		tok = isl_stream_next_token(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok, "expecting coefficient");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		if (tok->on_new_line) {
			isl_stream_error(s, tok,
				"coefficient should not appear on new line");
			isl_stream_push_token(s, tok);
			goto error;
		}
		pos = polylib_pos_to_isl_pos(bmap, j);
		if (pos >= 0)
			isl_int_set(c[pos], tok->u.v);
		isl_token_free(tok);
		if (pos < 0)
			return isl_basic_map_free(bmap);
	}

	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

static __isl_give isl_basic_map *basic_map_read_polylib(
	__isl_keep isl_stream *s)
{
	int i;
	struct isl_token *tok;
	struct isl_token *tok2;
	int n_row, n_col;
	int on_new_line;
	unsigned in = 0, out, local = 0;
	struct isl_basic_map *bmap = NULL;
	int nparam = 0;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	tok2 = isl_stream_next_token(s);
	if (!tok2) {
		isl_token_free(tok);
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	if (tok->type != ISL_TOKEN_VALUE || tok2->type != ISL_TOKEN_VALUE) {
		isl_stream_push_token(s, tok2);
		isl_stream_push_token(s, tok);
		isl_stream_error(s, NULL,
				 "expecting constraint matrix dimensions");
		return NULL;
	}
	n_row = isl_int_get_si(tok->u.v);
	n_col = isl_int_get_si(tok2->u.v);
	on_new_line = tok2->on_new_line;
	isl_token_free(tok2);
	isl_token_free(tok);
	isl_assert(s->ctx, !on_new_line, return NULL);
	isl_assert(s->ctx, n_row >= 0, return NULL);
	isl_assert(s->ctx, n_col >= 2 + nparam, return NULL);
	tok = isl_stream_next_token_on_same_line(s);
	if (tok) {
		if (tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok,
				    "expecting number of output dimensions");
			isl_stream_push_token(s, tok);
			goto error;
		}
		out = isl_int_get_si(tok->u.v);
		isl_token_free(tok);

		tok = isl_stream_next_token_on_same_line(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok,
				    "expecting number of input dimensions");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		in = isl_int_get_si(tok->u.v);
		isl_token_free(tok);

		tok = isl_stream_next_token_on_same_line(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok,
				    "expecting number of existentials");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		local = isl_int_get_si(tok->u.v);
		isl_token_free(tok);

		tok = isl_stream_next_token_on_same_line(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok,
				    "expecting number of parameters");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		nparam = isl_int_get_si(tok->u.v);
		isl_token_free(tok);
		if (n_col != 1 + out + in + local + nparam + 1) {
			isl_stream_error(s, NULL,
				    "dimensions don't match");
			goto error;
		}
	} else
		out = n_col - 2 - nparam;
	bmap = isl_basic_map_alloc(s->ctx, nparam, in, out, local, n_row, n_row);
	if (!bmap)
		return NULL;

	for (i = 0; i < local; ++i) {
		int k = isl_basic_map_alloc_div(bmap);
		if (k < 0)
			goto error;
		isl_seq_clr(bmap->div[k], 1 + 1 + nparam + in + out + local);
	}

	for (i = 0; i < n_row; ++i)
		bmap = basic_map_read_polylib_constraint(s, bmap);

	tok = isl_stream_next_token_on_same_line(s);
	if (tok) {
		isl_stream_error(s, tok, "unexpected extra token on line");
		isl_stream_push_token(s, tok);
		goto error;
	}

	bmap = isl_basic_map_simplify(bmap);
	bmap = isl_basic_map_finalize(bmap);
	return bmap;
error:
	isl_basic_map_free(bmap);
	return NULL;
}

static __isl_give isl_map *map_read_polylib(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	struct isl_token *tok2;
	int i, n;
	struct isl_map *map;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	tok2 = isl_stream_next_token_on_same_line(s);
	if (tok2 && tok2->type == ISL_TOKEN_VALUE) {
		isl_stream_push_token(s, tok2);
		isl_stream_push_token(s, tok);
		return isl_map_from_basic_map(basic_map_read_polylib(s));
	}
	if (tok2) {
		isl_stream_error(s, tok2, "unexpected token");
		isl_stream_push_token(s, tok2);
		isl_stream_push_token(s, tok);
		return NULL;
	}
	n = isl_int_get_si(tok->u.v);
	isl_token_free(tok);

	isl_assert(s->ctx, n >= 1, return NULL);

	map = isl_map_from_basic_map(basic_map_read_polylib(s));

	for (i = 1; map && i < n; ++i)
		map = isl_map_union(map,
			isl_map_from_basic_map(basic_map_read_polylib(s)));

	return map;
}

static int optional_power(__isl_keep isl_stream *s)
{
	int pow;
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 1;
	if (tok->type != '^') {
		isl_stream_push_token(s, tok);
		return 1;
	}
	isl_token_free(tok);
	tok = isl_stream_next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting exponent");
		if (tok)
			isl_stream_push_token(s, tok);
		return 1;
	}
	pow = isl_int_get_si(tok->u.v);
	isl_token_free(tok);
	return pow;
}

static __isl_give isl_pw_qpolynomial *read_term(__isl_keep isl_stream *s,
	__isl_keep isl_map *map, struct vars *v);

static __isl_give isl_pw_qpolynomial *read_factor(__isl_keep isl_stream *s,
	__isl_keep isl_map *map, struct vars *v)
{
	isl_pw_qpolynomial *pwqp;
	struct isl_token *tok;

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	if (tok->type == '(') {
		int pow;

		isl_token_free(tok);
		pwqp = read_term(s, map, v);
		if (!pwqp)
			return NULL;
		if (isl_stream_eat(s, ')'))
			goto error;
		pow = optional_power(s);
		pwqp = isl_pw_qpolynomial_pow(pwqp, pow);
	} else if (tok->type == ISL_TOKEN_VALUE) {
		struct isl_token *tok2;
		isl_qpolynomial *qp;

		tok2 = isl_stream_next_token(s);
		if (tok2 && tok2->type == '/') {
			isl_token_free(tok2);
			tok2 = next_token(s);
			if (!tok2 || tok2->type != ISL_TOKEN_VALUE) {
				isl_stream_error(s, tok2, "expected denominator");
				isl_token_free(tok);
				isl_token_free(tok2);
				return NULL;
			}
			qp = isl_qpolynomial_rat_cst_on_domain(isl_map_get_space(map),
						    tok->u.v, tok2->u.v);
			isl_token_free(tok2);
		} else {
			isl_stream_push_token(s, tok2);
			qp = isl_qpolynomial_cst_on_domain(isl_map_get_space(map),
						tok->u.v);
		}
		isl_token_free(tok);
		pwqp = isl_pw_qpolynomial_from_qpolynomial(qp);
	} else if (tok->type == ISL_TOKEN_INFTY) {
		isl_qpolynomial *qp;
		isl_token_free(tok);
		qp = isl_qpolynomial_infty_on_domain(isl_map_get_space(map));
		pwqp = isl_pw_qpolynomial_from_qpolynomial(qp);
	} else if (tok->type == ISL_TOKEN_NAN) {
		isl_qpolynomial *qp;
		isl_token_free(tok);
		qp = isl_qpolynomial_nan_on_domain(isl_map_get_space(map));
		pwqp = isl_pw_qpolynomial_from_qpolynomial(qp);
	} else if (tok->type == ISL_TOKEN_IDENT) {
		int n = v->n;
		int pos = vars_pos(v, tok->u.s, -1);
		int pow;
		isl_qpolynomial *qp;
		if (pos < 0) {
			isl_token_free(tok);
			return NULL;
		}
		if (pos >= n) {
			vars_drop(v, v->n - n);
			isl_stream_error(s, tok, "unknown identifier");
			isl_token_free(tok);
			return NULL;
		}
		isl_token_free(tok);
		pow = optional_power(s);
		qp = isl_qpolynomial_var_pow_on_domain(isl_map_get_space(map), pos, pow);
		pwqp = isl_pw_qpolynomial_from_qpolynomial(qp);
	} else if (is_start_of_div(tok)) {
		isl_pw_aff *pwaff;
		int pow;

		isl_stream_push_token(s, tok);
		pwaff = accept_div(s, isl_map_get_space(map), v);
		pow = optional_power(s);
		pwqp = isl_pw_qpolynomial_from_pw_aff(pwaff);
		pwqp = isl_pw_qpolynomial_pow(pwqp, pow);
	} else if (tok->type == '-') {
		isl_token_free(tok);
		pwqp = read_factor(s, map, v);
		pwqp = isl_pw_qpolynomial_neg(pwqp);
	} else {
		isl_stream_error(s, tok, "unexpected isl_token");
		isl_stream_push_token(s, tok);
		return NULL;
	}

	if (isl_stream_eat_if_available(s, '*') ||
	    isl_stream_next_token_is(s, ISL_TOKEN_IDENT)) {
		isl_pw_qpolynomial *pwqp2;

		pwqp2 = read_factor(s, map, v);
		pwqp = isl_pw_qpolynomial_mul(pwqp, pwqp2);
	}

	return pwqp;
error:
	isl_pw_qpolynomial_free(pwqp);
	return NULL;
}

static __isl_give isl_pw_qpolynomial *read_term(__isl_keep isl_stream *s,
	__isl_keep isl_map *map, struct vars *v)
{
	struct isl_token *tok;
	isl_pw_qpolynomial *pwqp;

	pwqp = read_factor(s, map, v);

	for (;;) {
		tok = next_token(s);
		if (!tok)
			return pwqp;

		if (tok->type == '+') {
			isl_pw_qpolynomial *pwqp2;

			isl_token_free(tok);
			pwqp2 = read_factor(s, map, v);
			pwqp = isl_pw_qpolynomial_add(pwqp, pwqp2);
		} else if (tok->type == '-') {
			isl_pw_qpolynomial *pwqp2;

			isl_token_free(tok);
			pwqp2 = read_factor(s, map, v);
			pwqp = isl_pw_qpolynomial_sub(pwqp, pwqp2);
		} else if (tok->type == ISL_TOKEN_VALUE &&
			    isl_int_is_neg(tok->u.v)) {
			isl_pw_qpolynomial *pwqp2;

			isl_stream_push_token(s, tok);
			pwqp2 = read_factor(s, map, v);
			pwqp = isl_pw_qpolynomial_add(pwqp, pwqp2);
		} else {
			isl_stream_push_token(s, tok);
			break;
		}
	}

	return pwqp;
}

static __isl_give isl_map *read_optional_formula(__isl_keep isl_stream *s,
	__isl_take isl_map *map, struct vars *v, int rational)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		goto error;
	}
	if (tok->type == ':' ||
	    (tok->type == ISL_TOKEN_OR && !strcmp(tok->u.s, "|"))) {
		isl_token_free(tok);
		map = read_formula(s, v, map, rational);
	} else
		isl_stream_push_token(s, tok);

	return map;
error:
	isl_map_free(map);
	return NULL;
}

static struct isl_obj obj_read_poly(__isl_keep isl_stream *s,
	__isl_take isl_map *map, struct vars *v, int n)
{
	struct isl_obj obj = { isl_obj_pw_qpolynomial, NULL };
	isl_pw_qpolynomial *pwqp;
	struct isl_set *set;

	pwqp = read_term(s, map, v);
	map = read_optional_formula(s, map, v, 0);
	set = isl_map_range(map);

	pwqp = isl_pw_qpolynomial_intersect_domain(pwqp, set);

	vars_drop(v, v->n - n);

	obj.v = pwqp;
	return obj;
}

static struct isl_obj obj_read_poly_or_fold(__isl_keep isl_stream *s,
	__isl_take isl_set *set, struct vars *v, int n)
{
	struct isl_obj obj = { isl_obj_pw_qpolynomial_fold, NULL };
	isl_pw_qpolynomial *pwqp;
	isl_pw_qpolynomial_fold *pwf = NULL;

	if (!isl_stream_eat_if_available(s, ISL_TOKEN_MAX))
		return obj_read_poly(s, set, v, n);

	if (isl_stream_eat(s, '('))
		goto error;

	pwqp = read_term(s, set, v);
	pwf = isl_pw_qpolynomial_fold_from_pw_qpolynomial(isl_fold_max, pwqp);

	while (isl_stream_eat_if_available(s, ',')) {
		isl_pw_qpolynomial_fold *pwf_i;
		pwqp = read_term(s, set, v);
		pwf_i = isl_pw_qpolynomial_fold_from_pw_qpolynomial(isl_fold_max,
									pwqp);
		pwf = isl_pw_qpolynomial_fold_fold(pwf, pwf_i);
	}

	if (isl_stream_eat(s, ')'))
		goto error;

	set = read_optional_formula(s, set, v, 0);
	pwf = isl_pw_qpolynomial_fold_intersect_domain(pwf, set);

	vars_drop(v, v->n - n);

	obj.v = pwf;
	return obj;
error:
	isl_set_free(set);
	isl_pw_qpolynomial_fold_free(pwf);
	obj.type = isl_obj_none;
	return obj;
}

static int is_rational(__isl_keep isl_stream *s)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type == ISL_TOKEN_RAT && isl_stream_next_token_is(s, ':')) {
		isl_token_free(tok);
		isl_stream_eat(s, ':');
		return 1;
	}

	isl_stream_push_token(s, tok);

	return 0;
}

static struct isl_obj obj_read_body(__isl_keep isl_stream *s,
	__isl_take isl_map *map, struct vars *v)
{
	struct isl_token *tok;
	struct isl_obj obj = { isl_obj_set, NULL };
	int n = v->n;
	int rational;

	rational = is_rational(s);
	if (rational)
		map = isl_map_set_rational(map);

	if (isl_stream_next_token_is(s, ':')) {
		obj.type = isl_obj_set;
		obj.v = read_optional_formula(s, map, v, rational);
		return obj;
	}

	if (!next_is_tuple(s))
		return obj_read_poly_or_fold(s, map, v, n);

	map = read_map_tuple(s, map, isl_dim_in, v, rational, 0);
	if (!map)
		goto error;
	tok = isl_stream_next_token(s);
	if (!tok)
		goto error;
	if (tok->type == ISL_TOKEN_TO) {
		obj.type = isl_obj_map;
		isl_token_free(tok);
		if (!next_is_tuple(s)) {
			isl_set *set = isl_map_domain(map);
			return obj_read_poly_or_fold(s, set, v, n);
		}
		map = read_map_tuple(s, map, isl_dim_out, v, rational, 0);
		if (!map)
			goto error;
	} else {
		map = isl_map_domain(map);
		isl_stream_push_token(s, tok);
	}

	map = read_optional_formula(s, map, v, rational);

	vars_drop(v, v->n - n);

	obj.v = map;
	return obj;
error:
	isl_map_free(map);
	obj.type = isl_obj_none;
	return obj;
}

static struct isl_obj to_union(isl_ctx *ctx, struct isl_obj obj)
{
	if (obj.type == isl_obj_map) {
		obj.v = isl_union_map_from_map(obj.v);
		obj.type = isl_obj_union_map;
	} else if (obj.type == isl_obj_set) {
		obj.v = isl_union_set_from_set(obj.v);
		obj.type = isl_obj_union_set;
	} else if (obj.type == isl_obj_pw_qpolynomial) {
		obj.v = isl_union_pw_qpolynomial_from_pw_qpolynomial(obj.v);
		obj.type = isl_obj_union_pw_qpolynomial;
	} else if (obj.type == isl_obj_pw_qpolynomial_fold) {
		obj.v = isl_union_pw_qpolynomial_fold_from_pw_qpolynomial_fold(obj.v);
		obj.type = isl_obj_union_pw_qpolynomial_fold;
	} else
		isl_assert(ctx, 0, goto error);
	return obj;
error:
	obj.type->free(obj.v);
	obj.type = isl_obj_none;
	return obj;
}

static struct isl_obj obj_add(__isl_keep isl_stream *s,
	struct isl_obj obj1, struct isl_obj obj2)
{
	if (obj2.type == isl_obj_none || !obj2.v)
		goto error;
	if (obj1.type == isl_obj_set && obj2.type == isl_obj_union_set)
		obj1 = to_union(s->ctx, obj1);
	if (obj1.type == isl_obj_union_set && obj2.type == isl_obj_set)
		obj2 = to_union(s->ctx, obj2);
	if (obj1.type == isl_obj_map && obj2.type == isl_obj_union_map)
		obj1 = to_union(s->ctx, obj1);
	if (obj1.type == isl_obj_union_map && obj2.type == isl_obj_map)
		obj2 = to_union(s->ctx, obj2);
	if (obj1.type == isl_obj_pw_qpolynomial &&
	    obj2.type == isl_obj_union_pw_qpolynomial)
		obj1 = to_union(s->ctx, obj1);
	if (obj1.type == isl_obj_union_pw_qpolynomial &&
	    obj2.type == isl_obj_pw_qpolynomial)
		obj2 = to_union(s->ctx, obj2);
	if (obj1.type == isl_obj_pw_qpolynomial_fold &&
	    obj2.type == isl_obj_union_pw_qpolynomial_fold)
		obj1 = to_union(s->ctx, obj1);
	if (obj1.type == isl_obj_union_pw_qpolynomial_fold &&
	    obj2.type == isl_obj_pw_qpolynomial_fold)
		obj2 = to_union(s->ctx, obj2);
	if (obj1.type != obj2.type) {
		isl_stream_error(s, NULL,
				"attempt to combine incompatible objects");
		goto error;
	}
	if (!obj1.type->add)
		isl_die(s->ctx, isl_error_internal,
			"combination not supported on object type", goto error);
	if (obj1.type == isl_obj_map && !isl_map_has_equal_space(obj1.v, obj2.v)) {
		obj1 = to_union(s->ctx, obj1);
		obj2 = to_union(s->ctx, obj2);
	}
	if (obj1.type == isl_obj_set && !isl_set_has_equal_space(obj1.v, obj2.v)) {
		obj1 = to_union(s->ctx, obj1);
		obj2 = to_union(s->ctx, obj2);
	}
	if (obj1.type == isl_obj_pw_qpolynomial &&
	    !isl_pw_qpolynomial_has_equal_space(obj1.v, obj2.v)) {
		obj1 = to_union(s->ctx, obj1);
		obj2 = to_union(s->ctx, obj2);
	}
	if (obj1.type == isl_obj_pw_qpolynomial_fold &&
	    !isl_pw_qpolynomial_fold_has_equal_space(obj1.v, obj2.v)) {
		obj1 = to_union(s->ctx, obj1);
		obj2 = to_union(s->ctx, obj2);
	}
	obj1.v = obj1.type->add(obj1.v, obj2.v);
	return obj1;
error:
	obj1.type->free(obj1.v);
	obj2.type->free(obj2.v);
	obj1.type = isl_obj_none;
	obj1.v = NULL;
	return obj1;
}

/* Are the first two tokens on "s", "domain" (either as a string
 * or as an identifier) followed by ":"?
 */
static int next_is_domain_colon(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	char *name;
	int res;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type != ISL_TOKEN_IDENT && tok->type != ISL_TOKEN_STRING) {
		isl_stream_push_token(s, tok);
		return 0;
	}

	name = isl_token_get_str(s->ctx, tok);
	res = !strcmp(name, "domain") && isl_stream_next_token_is(s, ':');
	free(name);

	isl_stream_push_token(s, tok);

	return res;
}

/* Do the first tokens on "s" look like a schedule?
 *
 * The root of a schedule is always a domain node, so the first thing
 * we expect in the stream is a domain key, i.e., "domain" followed
 * by ":".  If the schedule was printed in YAML flow style, then
 * we additionally expect a "{" to open the outer mapping.
 */
static int next_is_schedule(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	int is_schedule;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type != '{') {
		isl_stream_push_token(s, tok);
		return next_is_domain_colon(s);
	}

	is_schedule = next_is_domain_colon(s);
	isl_stream_push_token(s, tok);

	return is_schedule;
}

/* Read an isl_schedule from "s" and store it in an isl_obj.
 */
static struct isl_obj schedule_read(__isl_keep isl_stream *s)
{
	struct isl_obj obj;

	obj.type = isl_obj_schedule;
	obj.v = isl_stream_read_schedule(s);

	return obj;
}

/* Read a disjunction of object bodies from "s".
 * That is, read the inside of the braces, but not the braces themselves.
 * "v" contains a description of the identifiers parsed so far.
 * "map" contains information about the parameters.
 */
static struct isl_obj obj_read_disjuncts(__isl_keep isl_stream *s,
	struct vars *v, __isl_keep isl_map *map)
{
	struct isl_obj obj = { isl_obj_set, NULL };

	if (isl_stream_next_token_is(s, '}')) {
		obj.type = isl_obj_union_set;
		obj.v = isl_union_set_empty(isl_map_get_space(map));
		return obj;
	}

	for (;;) {
		struct isl_obj o;
		o = obj_read_body(s, isl_map_copy(map), v);
		if (!obj.v)
			obj = o;
		else
			obj = obj_add(s, obj, o);
		if (obj.type == isl_obj_none || !obj.v)
			return obj;
		if (!isl_stream_eat_if_available(s, ';'))
			break;
		if (isl_stream_next_token_is(s, '}'))
			break;
	}

	return obj;
}

static struct isl_obj obj_read(__isl_keep isl_stream *s)
{
	isl_map *map = NULL;
	struct isl_token *tok;
	struct vars *v = NULL;
	struct isl_obj obj = { isl_obj_set, NULL };

	if (next_is_schedule(s))
		return schedule_read(s);

	tok = next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		goto error;
	}
	if (tok->type == ISL_TOKEN_VALUE) {
		struct isl_token *tok2;
		struct isl_map *map;

		tok2 = isl_stream_next_token(s);
		if (!tok2 || tok2->type != ISL_TOKEN_VALUE ||
		    isl_int_is_neg(tok2->u.v)) {
			if (tok2)
				isl_stream_push_token(s, tok2);
			obj.type = isl_obj_val;
			obj.v = isl_val_int_from_isl_int(s->ctx, tok->u.v);
			isl_token_free(tok);
			return obj;
		}
		isl_stream_push_token(s, tok2);
		isl_stream_push_token(s, tok);
		map = map_read_polylib(s);
		if (!map)
			goto error;
		if (isl_map_may_be_set(map))
			obj.v = isl_map_range(map);
		else {
			obj.type = isl_obj_map;
			obj.v = map;
		}
		return obj;
	}
	v = vars_new(s->ctx);
	if (!v) {
		isl_stream_push_token(s, tok);
		goto error;
	}
	map = isl_map_universe(isl_space_params_alloc(s->ctx, 0));
	if (tok->type == '[') {
		isl_stream_push_token(s, tok);
		map = read_map_tuple(s, map, isl_dim_param, v, 0, 0);
		if (!map)
			goto error;
		tok = isl_stream_next_token(s);
		if (!tok || tok->type != ISL_TOKEN_TO) {
			isl_stream_error(s, tok, "expecting '->'");
			if (tok)
				isl_stream_push_token(s, tok);
			goto error;
		}
		isl_token_free(tok);
		tok = isl_stream_next_token(s);
	}
	if (!tok || tok->type != '{') {
		isl_stream_error(s, tok, "expecting '{'");
		if (tok)
			isl_stream_push_token(s, tok);
		goto error;
	}
	isl_token_free(tok);

	tok = isl_stream_next_token(s);
	if (!tok)
		;
	else if (tok->type == ISL_TOKEN_IDENT && !strcmp(tok->u.s, "Sym")) {
		isl_token_free(tok);
		if (isl_stream_eat(s, '='))
			goto error;
		map = read_map_tuple(s, map, isl_dim_param, v, 0, 1);
		if (!map)
			goto error;
	} else
		isl_stream_push_token(s, tok);

	obj = obj_read_disjuncts(s, v, map);
	if (obj.type == isl_obj_none || !obj.v)
		goto error;

	tok = isl_stream_next_token(s);
	if (tok && tok->type == '}') {
		isl_token_free(tok);
	} else {
		isl_stream_error(s, tok, "unexpected isl_token");
		if (tok)
			isl_token_free(tok);
		goto error;
	}

	vars_free(v);
	isl_map_free(map);

	return obj;
error:
	isl_map_free(map);
	obj.type->free(obj.v);
	if (v)
		vars_free(v);
	obj.v = NULL;
	return obj;
}

struct isl_obj isl_stream_read_obj(__isl_keep isl_stream *s)
{
	return obj_read(s);
}

__isl_give isl_map *isl_stream_read_map(__isl_keep isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.v)
		isl_assert(s->ctx, obj.type == isl_obj_map ||
				   obj.type == isl_obj_set, goto error);
	
	if (obj.type == isl_obj_set)
		obj.v = isl_map_from_range(obj.v);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_set *isl_stream_read_set(__isl_keep isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.v) {
		if (obj.type == isl_obj_map && isl_map_may_be_set(obj.v)) {
			obj.v = isl_map_range(obj.v);
			obj.type = isl_obj_set;
		}
		isl_assert(s->ctx, obj.type == isl_obj_set, goto error);
	}

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_union_map *isl_stream_read_union_map(__isl_keep isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.type == isl_obj_map) {
		obj.type = isl_obj_union_map;
		obj.v = isl_union_map_from_map(obj.v);
	}
	if (obj.type == isl_obj_set) {
		obj.type = isl_obj_union_set;
		obj.v = isl_union_set_from_set(obj.v);
	}
	if (obj.v && obj.type == isl_obj_union_set &&
	    isl_union_set_is_empty(obj.v))
		obj.type = isl_obj_union_map;
	if (obj.v && obj.type != isl_obj_union_map)
		isl_die(s->ctx, isl_error_invalid, "invalid input", goto error);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

/* Extract an isl_union_set from "obj".
 * This only works if the object was detected as either a set
 * (in which case it is converted to a union set) or a union set.
 */
static __isl_give isl_union_set *extract_union_set(isl_ctx *ctx,
	struct isl_obj obj)
{
	if (obj.type == isl_obj_set) {
		obj.type = isl_obj_union_set;
		obj.v = isl_union_set_from_set(obj.v);
	}
	if (obj.v)
		isl_assert(ctx, obj.type == isl_obj_union_set, goto error);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

/* Read an isl_union_set from "s".
 * First read a generic object and then try and extract
 * an isl_union_set from that.
 */
__isl_give isl_union_set *isl_stream_read_union_set(__isl_keep isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	return extract_union_set(s->ctx, obj);
}

static __isl_give isl_basic_map *basic_map_read(__isl_keep isl_stream *s)
{
	struct isl_obj obj;
	struct isl_map *map;
	struct isl_basic_map *bmap;

	obj = obj_read(s);
	if (obj.v && (obj.type != isl_obj_map && obj.type != isl_obj_set))
		isl_die(s->ctx, isl_error_invalid, "not a (basic) set or map",
			goto error);
	map = obj.v;
	if (!map)
		return NULL;

	if (map->n > 1)
		isl_die(s->ctx, isl_error_invalid,
			"set or map description involves "
			"more than one disjunct", goto error);

	if (map->n == 0)
		bmap = isl_basic_map_empty(isl_map_get_space(map));
	else
		bmap = isl_basic_map_copy(map->p[0]);

	isl_map_free(map);

	return bmap;
error:
	obj.type->free(obj.v);
	return NULL;
}

static __isl_give isl_basic_set *basic_set_read(__isl_keep isl_stream *s)
{
	isl_basic_map *bmap;
	bmap = basic_map_read(s);
	if (!bmap)
		return NULL;
	if (!isl_basic_map_may_be_set(bmap))
		isl_die(s->ctx, isl_error_invalid,
			"input is not a set", goto error);
	return isl_basic_map_range(bmap);
error:
	isl_basic_map_free(bmap);
	return NULL;
}

__isl_give isl_basic_map *isl_basic_map_read_from_file(isl_ctx *ctx,
	FILE *input)
{
	struct isl_basic_map *bmap;
	isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	bmap = basic_map_read(s);
	isl_stream_free(s);
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_read_from_file(isl_ctx *ctx,
	FILE *input)
{
	isl_basic_set *bset;
	isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	bset = basic_set_read(s);
	isl_stream_free(s);
	return bset;
}

__isl_give isl_basic_map *isl_basic_map_read_from_str(isl_ctx *ctx,
	const char *str)
{
	struct isl_basic_map *bmap;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	bmap = basic_map_read(s);
	isl_stream_free(s);
	return bmap;
}

__isl_give isl_basic_set *isl_basic_set_read_from_str(isl_ctx *ctx,
	const char *str)
{
	isl_basic_set *bset;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	bset = basic_set_read(s);
	isl_stream_free(s);
	return bset;
}

__isl_give isl_map *isl_map_read_from_file(struct isl_ctx *ctx,
	FILE *input)
{
	struct isl_map *map;
	isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	map = isl_stream_read_map(s);
	isl_stream_free(s);
	return map;
}

__isl_give isl_map *isl_map_read_from_str(struct isl_ctx *ctx,
	const char *str)
{
	struct isl_map *map;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	map = isl_stream_read_map(s);
	isl_stream_free(s);
	return map;
}

__isl_give isl_set *isl_set_read_from_file(struct isl_ctx *ctx,
	FILE *input)
{
	isl_set *set;
	isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	set = isl_stream_read_set(s);
	isl_stream_free(s);
	return set;
}

__isl_give isl_set *isl_set_read_from_str(isl_ctx *ctx, const char *str)
{
	isl_set *set;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	set = isl_stream_read_set(s);
	isl_stream_free(s);
	return set;
}

__isl_give isl_union_map *isl_union_map_read_from_file(isl_ctx *ctx,
	FILE *input)
{
	isl_union_map *umap;
	isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	umap = isl_stream_read_union_map(s);
	isl_stream_free(s);
	return umap;
}

__isl_give isl_union_map *isl_union_map_read_from_str(struct isl_ctx *ctx,
		const char *str)
{
	isl_union_map *umap;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	umap = isl_stream_read_union_map(s);
	isl_stream_free(s);
	return umap;
}

__isl_give isl_union_set *isl_union_set_read_from_file(isl_ctx *ctx,
	FILE *input)
{
	isl_union_set *uset;
	isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	uset = isl_stream_read_union_set(s);
	isl_stream_free(s);
	return uset;
}

__isl_give isl_union_set *isl_union_set_read_from_str(struct isl_ctx *ctx,
		const char *str)
{
	isl_union_set *uset;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	uset = isl_stream_read_union_set(s);
	isl_stream_free(s);
	return uset;
}

static __isl_give isl_vec *isl_vec_read_polylib(__isl_keep isl_stream *s)
{
	struct isl_vec *vec = NULL;
	struct isl_token *tok;
	unsigned size;
	int j;

	tok = isl_stream_next_token(s);
	if (!tok || tok->type != ISL_TOKEN_VALUE) {
		isl_stream_error(s, tok, "expecting vector length");
		goto error;
	}

	size = isl_int_get_si(tok->u.v);
	isl_token_free(tok);

	vec = isl_vec_alloc(s->ctx, size);

	for (j = 0; j < size; ++j) {
		tok = isl_stream_next_token(s);
		if (!tok || tok->type != ISL_TOKEN_VALUE) {
			isl_stream_error(s, tok, "expecting constant value");
			goto error;
		}
		isl_int_set(vec->el[j], tok->u.v);
		isl_token_free(tok);
	}

	return vec;
error:
	isl_token_free(tok);
	isl_vec_free(vec);
	return NULL;
}

static __isl_give isl_vec *vec_read(__isl_keep isl_stream *s)
{
	return isl_vec_read_polylib(s);
}

__isl_give isl_vec *isl_vec_read_from_file(isl_ctx *ctx, FILE *input)
{
	isl_vec *v;
	isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	v = vec_read(s);
	isl_stream_free(s);
	return v;
}

__isl_give isl_pw_qpolynomial *isl_stream_read_pw_qpolynomial(
	__isl_keep isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.v)
		isl_assert(s->ctx, obj.type == isl_obj_pw_qpolynomial,
			   goto error);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_read_from_str(isl_ctx *ctx,
		const char *str)
{
	isl_pw_qpolynomial *pwqp;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	pwqp = isl_stream_read_pw_qpolynomial(s);
	isl_stream_free(s);
	return pwqp;
}

__isl_give isl_pw_qpolynomial *isl_pw_qpolynomial_read_from_file(isl_ctx *ctx,
		FILE *input)
{
	isl_pw_qpolynomial *pwqp;
	isl_stream *s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	pwqp = isl_stream_read_pw_qpolynomial(s);
	isl_stream_free(s);
	return pwqp;
}

/* Is the next token an identifer not in "v"?
 */
static int next_is_fresh_ident(__isl_keep isl_stream *s, struct vars *v)
{
	int n = v->n;
	int fresh;
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	fresh = tok->type == ISL_TOKEN_IDENT && vars_pos(v, tok->u.s, -1) >= n;
	isl_stream_push_token(s, tok);

	vars_drop(v, v->n - n);

	return fresh;
}

/* First read the domain of the affine expression, which may be
 * a parameter space or a set.
 * The tricky part is that we don't know if the domain is a set or not,
 * so when we are trying to read the domain, we may actually be reading
 * the affine expression itself (defined on a parameter domains)
 * If the tuple we are reading is named, we assume it's the domain.
 * Also, if inside the tuple, the first thing we find is a nested tuple
 * or a new identifier, we again assume it's the domain.
 * Finally, if the tuple is empty, then it must be the domain
 * since it does not contain an affine expression.
 * Otherwise, we assume we are reading an affine expression.
 */
static __isl_give isl_set *read_aff_domain(__isl_keep isl_stream *s,
	__isl_take isl_set *dom, struct vars *v)
{
	struct isl_token *tok, *tok2;
	int is_empty;

	tok = isl_stream_next_token(s);
	if (tok && (tok->type == ISL_TOKEN_IDENT || tok->is_keyword)) {
		isl_stream_push_token(s, tok);
		return read_map_tuple(s, dom, isl_dim_set, v, 0, 0);
	}
	if (!tok || tok->type != '[') {
		isl_stream_error(s, tok, "expecting '['");
		goto error;
	}
	tok2 = isl_stream_next_token(s);
	is_empty = tok2 && tok2->type == ']';
	if (tok2)
		isl_stream_push_token(s, tok2);
	if (is_empty || next_is_tuple(s) || next_is_fresh_ident(s, v)) {
		isl_stream_push_token(s, tok);
		dom = read_map_tuple(s, dom, isl_dim_set, v, 0, 0);
	} else
		isl_stream_push_token(s, tok);

	return dom;
error:
	if (tok)
		isl_stream_push_token(s, tok);
	isl_set_free(dom);
	return NULL;
}

/* Read an affine expression from "s".
 */
__isl_give isl_aff *isl_stream_read_aff(__isl_keep isl_stream *s)
{
	isl_aff *aff;
	isl_multi_aff *ma;
	isl_size dim;

	ma = isl_stream_read_multi_aff(s);
	dim = isl_multi_aff_dim(ma, isl_dim_out);
	if (dim < 0)
		goto error;
	if (dim != 1)
		isl_die(s->ctx, isl_error_invalid,
			"expecting single affine expression",
			goto error);

	aff = isl_multi_aff_get_aff(ma, 0);
	isl_multi_aff_free(ma);
	return aff;
error:
	isl_multi_aff_free(ma);
	return NULL;
}

/* Read a piecewise affine expression from "s" with domain (space) "dom".
 */
static __isl_give isl_pw_aff *read_pw_aff_with_dom(__isl_keep isl_stream *s,
	__isl_take isl_set *dom, struct vars *v)
{
	isl_pw_aff *pwaff = NULL;

	if (!isl_set_is_params(dom) && isl_stream_eat(s, ISL_TOKEN_TO))
		goto error;

	if (isl_stream_eat(s, '['))
		goto error;

	pwaff = accept_affine(s, isl_set_get_space(dom), v);

	if (isl_stream_eat(s, ']'))
		goto error;

	dom = read_optional_formula(s, dom, v, 0);
	pwaff = isl_pw_aff_intersect_domain(pwaff, dom);

	return pwaff;
error:
	isl_set_free(dom);
	isl_pw_aff_free(pwaff);
	return NULL;
}

__isl_give isl_pw_aff *isl_stream_read_pw_aff(__isl_keep isl_stream *s)
{
	struct vars *v;
	isl_set *dom = NULL;
	isl_set *aff_dom;
	isl_pw_aff *pa = NULL;
	int n;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (isl_stream_eat(s, '{'))
		goto error;

	n = v->n;
	aff_dom = read_aff_domain(s, isl_set_copy(dom), v);
	pa = read_pw_aff_with_dom(s, aff_dom, v);
	vars_drop(v, v->n - n);

	while (isl_stream_eat_if_available(s, ';')) {
		isl_pw_aff *pa_i;

		n = v->n;
		aff_dom = read_aff_domain(s, isl_set_copy(dom), v);
		pa_i = read_pw_aff_with_dom(s, aff_dom, v);
		vars_drop(v, v->n - n);

		pa = isl_pw_aff_union_add(pa, pa_i);
	}

	if (isl_stream_eat(s, '}'))
		goto error;

	vars_free(v);
	isl_set_free(dom);
	return pa;
error:
	vars_free(v);
	isl_set_free(dom);
	isl_pw_aff_free(pa);
	return NULL;
}

__isl_give isl_aff *isl_aff_read_from_str(isl_ctx *ctx, const char *str)
{
	isl_aff *aff;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	aff = isl_stream_read_aff(s);
	isl_stream_free(s);
	return aff;
}

__isl_give isl_pw_aff *isl_pw_aff_read_from_str(isl_ctx *ctx, const char *str)
{
	isl_pw_aff *pa;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	pa = isl_stream_read_pw_aff(s);
	isl_stream_free(s);
	return pa;
}

/* Extract an isl_multi_pw_aff with domain space "dom_space"
 * from a tuple "tuple" read by read_tuple.
 *
 * Note that the function read_tuple accepts tuples where some output or
 * set dimensions are defined in terms of other output or set dimensions
 * since this function is also used to read maps.  As a special case,
 * read_tuple also accept dimensions that are defined in terms of themselves
 * (i.e., that are not defined).
 * These cases are not allowed when extracting an isl_multi_pw_aff so check
 * that the definitions of the output/set dimensions do not involve any
 * output/set dimensions.
 * Finally, drop the output dimensions from the domain of the result
 * of read_tuple (which is of the form [input, output] -> [output],
 * with anonymous domain) and reset the space.
 */
static __isl_give isl_multi_pw_aff *extract_mpa_from_tuple(
	__isl_take isl_space *dom_space, __isl_keep isl_multi_pw_aff *tuple)
{
	int i;
	isl_size dim, n;
	isl_space *space;
	isl_multi_pw_aff *mpa;

	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	dim = isl_space_dim(dom_space, isl_dim_all);
	if (n < 0 || dim < 0)
		dom_space = isl_space_free(dom_space);
	space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
	space = isl_space_align_params(space, isl_space_copy(dom_space));
	if (!isl_space_is_params(dom_space))
		space = isl_space_map_from_domain_and_range(
				isl_space_copy(dom_space), space);
	isl_space_free(dom_space);
	mpa = isl_multi_pw_aff_alloc(space);

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		if (!pa)
			return isl_multi_pw_aff_free(mpa);
		if (isl_pw_aff_involves_dims(pa, isl_dim_in, dim, i + 1)) {
			isl_ctx *ctx = isl_pw_aff_get_ctx(pa);
			isl_pw_aff_free(pa);
			isl_die(ctx, isl_error_invalid,
				"not an affine expression",
				return isl_multi_pw_aff_free(mpa));
		}
		pa = isl_pw_aff_drop_dims(pa, isl_dim_in, dim, n);
		space = isl_multi_pw_aff_get_domain_space(mpa);
		pa = isl_pw_aff_reset_domain_space(pa, space);
		mpa = isl_multi_pw_aff_set_pw_aff(mpa, i, pa);
	}

	return mpa;
}

/* Read a tuple of affine expressions, together with optional constraints
 * on the domain from "s".  "dom" represents the initial constraints
 * on the domain.
 *
 * The isl_multi_aff may live in either a set or a map space.
 * First read the first tuple and check if it is followed by a "->".
 * If so, convert the tuple into the domain of the isl_multi_pw_aff and
 * read in the next tuple.  This tuple (or the first tuple if it was
 * not followed by a "->") is then converted into an isl_multi_pw_aff
 * through a call to extract_mpa_from_tuple.
 * The result is converted to an isl_pw_multi_aff and
 * its domain is intersected with the domain.
 */
static __isl_give isl_pw_multi_aff *read_conditional_multi_aff(
	__isl_keep isl_stream *s, __isl_take isl_set *dom, struct vars *v)
{
	isl_multi_pw_aff *tuple;
	isl_multi_pw_aff *mpa;
	isl_pw_multi_aff *pma;
	int n = v->n;

	tuple = read_tuple(s, v, 0, 0);
	if (!tuple)
		goto error;
	if (isl_stream_eat_if_available(s, ISL_TOKEN_TO)) {
		isl_map *map = map_from_tuple(tuple, dom, isl_dim_in, v, 0);
		dom = isl_map_domain(map);
		tuple = read_tuple(s, v, 0, 0);
		if (!tuple)
			goto error;
	}
	mpa = extract_mpa_from_tuple(isl_set_get_space(dom), tuple);
	isl_multi_pw_aff_free(tuple);
	if (!mpa)
		dom = isl_set_free(dom);

	dom = read_optional_formula(s, dom, v, 0);

	vars_drop(v, v->n - n);

	pma = isl_pw_multi_aff_from_multi_pw_aff(mpa);
	pma = isl_pw_multi_aff_intersect_domain(pma, dom);

	return pma;
error:
	isl_set_free(dom);
	return NULL;
}

/* Read an isl_union_pw_multi_aff from "s".
 *
 * In particular, first read the parameters and then read a sequence
 * of zero or more tuples of affine expressions with optional conditions and
 * add them up.
 */
__isl_give isl_union_pw_multi_aff *isl_stream_read_union_pw_multi_aff(
	__isl_keep isl_stream *s)
{
	struct vars *v;
	isl_set *dom;
	isl_union_pw_multi_aff *upma = NULL;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (isl_stream_eat(s, '{'))
		goto error;

	upma = isl_union_pw_multi_aff_empty(isl_set_get_space(dom));

	do {
		isl_pw_multi_aff *pma;
		isl_union_pw_multi_aff *upma2;

		if (isl_stream_next_token_is(s, '}'))
			break;

		pma = read_conditional_multi_aff(s, isl_set_copy(dom), v);
		upma2 = isl_union_pw_multi_aff_from_pw_multi_aff(pma);
		upma = isl_union_pw_multi_aff_union_add(upma, upma2);
		if (!upma)
			goto error;
	} while (isl_stream_eat_if_available(s, ';'));

	if (isl_stream_eat(s, '}'))
		goto error;

	isl_set_free(dom);
	vars_free(v);
	return upma;
error:
	isl_union_pw_multi_aff_free(upma);
	isl_set_free(dom);
	vars_free(v);
	return NULL;
}

/* Read an isl_pw_multi_aff from "s".
 *
 * Read a more generic isl_union_pw_multi_aff first and
 * then check that the result lives in a single space.
 */
__isl_give isl_pw_multi_aff *isl_stream_read_pw_multi_aff(
	__isl_keep isl_stream *s)
{
	isl_bool single_space;
	isl_union_pw_multi_aff *upma;

	upma = isl_stream_read_union_pw_multi_aff(s);
	single_space = isl_union_pw_multi_aff_isa_pw_multi_aff(upma);
	if (single_space < 0)
		upma = isl_union_pw_multi_aff_free(upma);
	else if (!single_space)
		isl_die(s->ctx, isl_error_invalid,
			"expecting expression in single space",
			upma = isl_union_pw_multi_aff_free(upma));
	return isl_union_pw_multi_aff_as_pw_multi_aff(upma);
}

__isl_give isl_pw_multi_aff *isl_pw_multi_aff_read_from_str(isl_ctx *ctx,
	const char *str)
{
	isl_pw_multi_aff *pma;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	pma = isl_stream_read_pw_multi_aff(s);
	isl_stream_free(s);
	return pma;
}

/* Read an isl_union_pw_multi_aff from "str".
 */
__isl_give isl_union_pw_multi_aff *isl_union_pw_multi_aff_read_from_str(
	isl_ctx *ctx, const char *str)
{
	isl_union_pw_multi_aff *upma;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	upma = isl_stream_read_union_pw_multi_aff(s);
	isl_stream_free(s);
	return upma;
}

/* Assuming "pa" represents a single affine expression defined on a universe
 * domain, extract this affine expression.
 */
static __isl_give isl_aff *aff_from_pw_aff(__isl_take isl_pw_aff *pa)
{
	isl_aff *aff;

	if (!pa)
		return NULL;
	if (pa->n != 1)
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"expecting single affine expression",
			goto error);
	if (!isl_set_plain_is_universe(pa->p[0].set))
		isl_die(isl_pw_aff_get_ctx(pa), isl_error_invalid,
			"expecting universe domain",
			goto error);

	aff = isl_aff_copy(pa->p[0].aff);
	isl_pw_aff_free(pa);
	return aff;
error:
	isl_pw_aff_free(pa);
	return NULL;
}

#undef BASE
#define BASE val

#include <isl_multi_read_no_explicit_domain_templ.c>

#undef BASE
#define BASE id

#include <isl_multi_read_no_explicit_domain_templ.c>

/* Read a multi-affine expression from "s".
 * If the multi-affine expression has a domain, then the tuple
 * representing this domain cannot involve any affine expressions.
 * The tuple representing the actual expressions needs to consist
 * of only affine expressions.  Moreover, these expressions can
 * only depend on parameters and input dimensions and not on other
 * output dimensions.
 */
__isl_give isl_multi_aff *isl_stream_read_multi_aff(__isl_keep isl_stream *s)
{
	struct vars *v;
	isl_set *dom = NULL;
	isl_multi_pw_aff *tuple = NULL;
	int i;
	isl_size dim, n;
	isl_space *space, *dom_space;
	isl_multi_aff *ma = NULL;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (!isl_set_plain_is_universe(dom))
		isl_die(s->ctx, isl_error_invalid,
			"expecting universe parameter domain", goto error);
	if (isl_stream_eat(s, '{'))
		goto error;

	tuple = read_tuple(s, v, 0, 0);
	if (!tuple)
		goto error;
	if (isl_stream_eat_if_available(s, ISL_TOKEN_TO)) {
		isl_set *set;
		isl_space *space;
		isl_bool has_expr;

		has_expr = tuple_has_expr(tuple);
		if (has_expr < 0)
			goto error;
		if (has_expr)
			isl_die(s->ctx, isl_error_invalid,
				"expecting universe domain", goto error);
		space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
		set = isl_set_universe(space);
		dom = isl_set_intersect_params(set, dom);
		isl_multi_pw_aff_free(tuple);
		tuple = read_tuple(s, v, 0, 0);
		if (!tuple)
			goto error;
	}

	if (isl_stream_eat(s, '}'))
		goto error;

	n = isl_multi_pw_aff_dim(tuple, isl_dim_out);
	dim = isl_set_dim(dom, isl_dim_all);
	if (n < 0 || dim < 0)
		goto error;
	dom_space = isl_set_get_space(dom);
	space = isl_space_range(isl_multi_pw_aff_get_space(tuple));
	space = isl_space_align_params(space, isl_space_copy(dom_space));
	if (!isl_space_is_params(dom_space))
		space = isl_space_map_from_domain_and_range(
				isl_space_copy(dom_space), space);
	isl_space_free(dom_space);
	ma = isl_multi_aff_alloc(space);

	for (i = 0; i < n; ++i) {
		isl_pw_aff *pa;
		isl_aff *aff;
		pa = isl_multi_pw_aff_get_pw_aff(tuple, i);
		aff = aff_from_pw_aff(pa);
		if (!aff)
			goto error;
		if (isl_aff_involves_dims(aff, isl_dim_in, dim, i + 1)) {
			isl_aff_free(aff);
			isl_die(s->ctx, isl_error_invalid,
				"not an affine expression", goto error);
		}
		aff = isl_aff_drop_dims(aff, isl_dim_in, dim, n);
		space = isl_multi_aff_get_domain_space(ma);
		aff = isl_aff_reset_domain_space(aff, space);
		ma = isl_multi_aff_set_aff(ma, i, aff);
	}

	isl_multi_pw_aff_free(tuple);
	vars_free(v);
	isl_set_free(dom);
	return ma;
error:
	isl_multi_pw_aff_free(tuple);
	vars_free(v);
	isl_set_free(dom);
	isl_multi_aff_free(ma);
	return NULL;
}

__isl_give isl_multi_aff *isl_multi_aff_read_from_str(isl_ctx *ctx,
	const char *str)
{
	isl_multi_aff *maff;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	maff = isl_stream_read_multi_aff(s);
	isl_stream_free(s);
	return maff;
}

/* Read an isl_multi_pw_aff from "s".
 *
 * The input format is similar to that of map, except that any conditions
 * on the domains should be specified inside the tuple since each
 * piecewise affine expression may have a different domain.
 * However, additional, shared conditions can also be specified.
 * This is especially useful for setting the explicit domain
 * of a zero-dimensional isl_multi_pw_aff.
 *
 * Since we do not know in advance if the isl_multi_pw_aff lives
 * in a set or a map space, we first read the first tuple and check
 * if it is followed by a "->".  If so, we convert the tuple into
 * the domain of the isl_multi_pw_aff and read in the next tuple.
 * This tuple (or the first tuple if it was not followed by a "->")
 * is then converted into the isl_multi_pw_aff through a call
 * to extract_mpa_from_tuple and the domain of the result
 * is intersected with the domain.
 */
__isl_give isl_multi_pw_aff *isl_stream_read_multi_pw_aff(
	__isl_keep isl_stream *s)
{
	struct vars *v;
	isl_set *dom = NULL;
	isl_multi_pw_aff *tuple = NULL;
	isl_multi_pw_aff *mpa = NULL;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (isl_stream_eat(s, '{'))
		goto error;

	tuple = read_tuple(s, v, 0, 0);
	if (!tuple)
		goto error;
	if (isl_stream_eat_if_available(s, ISL_TOKEN_TO)) {
		isl_map *map = map_from_tuple(tuple, dom, isl_dim_in, v, 0);
		dom = isl_map_domain(map);
		tuple = read_tuple(s, v, 0, 0);
		if (!tuple)
			goto error;
	}

	if (isl_stream_eat_if_available(s, ':'))
		dom = read_formula(s, v, dom, 0);

	if (isl_stream_eat(s, '}'))
		goto error;

	mpa = extract_mpa_from_tuple(isl_set_get_space(dom), tuple);

	isl_multi_pw_aff_free(tuple);
	vars_free(v);
	mpa = isl_multi_pw_aff_intersect_domain(mpa, dom);
	return mpa;
error:
	isl_multi_pw_aff_free(tuple);
	vars_free(v);
	isl_set_free(dom);
	isl_multi_pw_aff_free(mpa);
	return NULL;
}

/* Read an isl_multi_pw_aff from "str".
 */
__isl_give isl_multi_pw_aff *isl_multi_pw_aff_read_from_str(isl_ctx *ctx,
	const char *str)
{
	isl_multi_pw_aff *mpa;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	mpa = isl_stream_read_multi_pw_aff(s);
	isl_stream_free(s);
	return mpa;
}

/* Read the body of an isl_union_pw_aff from "s" with parameter domain "dom".
 */
static __isl_give isl_union_pw_aff *read_union_pw_aff_with_dom(
	__isl_keep isl_stream *s, __isl_take isl_set *dom, struct vars *v)
{
	isl_pw_aff *pa;
	isl_union_pw_aff *upa = NULL;
	isl_set *aff_dom;
	int n;

	n = v->n;
	aff_dom = read_aff_domain(s, isl_set_copy(dom), v);
	pa = read_pw_aff_with_dom(s, aff_dom, v);
	vars_drop(v, v->n - n);

	upa = isl_union_pw_aff_from_pw_aff(pa);

	while (isl_stream_eat_if_available(s, ';')) {
		isl_pw_aff *pa_i;
		isl_union_pw_aff *upa_i;

		n = v->n;
		aff_dom = read_aff_domain(s, isl_set_copy(dom), v);
		pa_i = read_pw_aff_with_dom(s, aff_dom, v);
		vars_drop(v, v->n - n);

		upa_i = isl_union_pw_aff_from_pw_aff(pa_i);
		upa = isl_union_pw_aff_union_add(upa, upa_i);
	}

	isl_set_free(dom);
	return upa;
}

/* Read an isl_union_pw_aff from "s".
 *
 * First check if there are any paramters, then read in the opening brace
 * and use read_union_pw_aff_with_dom to read in the body of
 * the isl_union_pw_aff.  Finally, read the closing brace.
 */
__isl_give isl_union_pw_aff *isl_stream_read_union_pw_aff(
	__isl_keep isl_stream *s)
{
	struct vars *v;
	isl_set *dom;
	isl_union_pw_aff *upa = NULL;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (isl_stream_eat(s, '{'))
		goto error;

	upa = read_union_pw_aff_with_dom(s, isl_set_copy(dom), v);

	if (isl_stream_eat(s, '}'))
		goto error;

	vars_free(v);
	isl_set_free(dom);
	return upa;
error:
	vars_free(v);
	isl_set_free(dom);
	isl_union_pw_aff_free(upa);
	return NULL;
}

/* Read an isl_union_pw_aff from "str".
 */
__isl_give isl_union_pw_aff *isl_union_pw_aff_read_from_str(isl_ctx *ctx,
	const char *str)
{
	isl_union_pw_aff *upa;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	upa = isl_stream_read_union_pw_aff(s);
	isl_stream_free(s);
	return upa;
}

/* This function is called for each element in a tuple inside
 * isl_stream_read_multi_union_pw_aff.
 *
 * Read a '{', the union piecewise affine expression body and a '}' and
 * add the isl_union_pw_aff to *list.
 */
static __isl_give isl_space *read_union_pw_aff_el(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_space *space, int rational, void *user)
{
	isl_set *dom;
	isl_union_pw_aff *upa;
	isl_union_pw_aff_list **list = (isl_union_pw_aff_list **) user;

	dom = isl_set_universe(isl_space_params(isl_space_copy(space)));
	if (isl_stream_eat(s, '{'))
		goto error;
	upa = read_union_pw_aff_with_dom(s, dom, v);
	*list = isl_union_pw_aff_list_add(*list, upa);
	if (isl_stream_eat(s, '}'))
		return isl_space_free(space);
	if (!*list)
		return isl_space_free(space);
	return space;
error:
	isl_set_free(dom);
	return isl_space_free(space);
}

/* Do the next tokens in "s" correspond to an empty tuple?
 * In particular, does the stream start with a '[', followed by a ']',
 * not followed by a "->"?
 */
static int next_is_empty_tuple(__isl_keep isl_stream *s)
{
	struct isl_token *tok, *tok2, *tok3;
	int is_empty_tuple = 0;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type != '[') {
		isl_stream_push_token(s, tok);
		return 0;
	}

	tok2 = isl_stream_next_token(s);
	if (tok2 && tok2->type == ']') {
		tok3 = isl_stream_next_token(s);
		is_empty_tuple = !tok || tok->type != ISL_TOKEN_TO;
		if (tok3)
			isl_stream_push_token(s, tok3);
	}
	if (tok2)
		isl_stream_push_token(s, tok2);
	isl_stream_push_token(s, tok);

	return is_empty_tuple;
}

/* Do the next tokens in "s" correspond to a tuple of parameters?
 * In particular, does the stream start with a '[' that is not
 * followed by a '{' or a nested tuple?
 */
static int next_is_param_tuple(__isl_keep isl_stream *s)
{
	struct isl_token *tok, *tok2;
	int is_tuple;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type != '[' || next_is_tuple(s)) {
		isl_stream_push_token(s, tok);
		return 0;
	}

	tok2 = isl_stream_next_token(s);
	is_tuple = tok2 && tok2->type != '{';
	if (tok2)
		isl_stream_push_token(s, tok2);
	isl_stream_push_token(s, tok);

	return is_tuple;
}

/* Read the core of a body of an isl_multi_union_pw_aff from "s",
 * i.e., everything except the parameter specification and
 * without shared domain constraints.
 * "v" contains a description of the identifiers parsed so far.
 * The parameters, if any, are specified by "space".
 *
 * The body is of the form
 *
 *	[{ [..] : ... ; [..] : ... }, { [..] : ... ; [..] : ... }]
 *
 * Read the tuple, collecting the individual isl_union_pw_aff
 * elements in a list and construct the result from the tuple space and
 * the list.
 */
static __isl_give isl_multi_union_pw_aff *read_multi_union_pw_aff_body_core(
	__isl_keep isl_stream *s, struct vars *v, __isl_take isl_space *space)
{
	isl_union_pw_aff_list *list;
	isl_multi_union_pw_aff *mupa;

	list = isl_union_pw_aff_list_alloc(s->ctx, 0);
	space = read_tuple_space(s, v, space, 1, 0,
				&read_union_pw_aff_el, &list);
	mupa = isl_multi_union_pw_aff_from_union_pw_aff_list(space, list);

	return mupa;
}

/* Read the body of an isl_union_set from "s",
 * i.e., everything except the parameter specification.
 * "v" contains a description of the identifiers parsed so far.
 * The parameters, if any, are specified by "space".
 *
 * First read a generic disjunction of object bodies and then try and extract
 * an isl_union_set from that.
 */
static __isl_give isl_union_set *read_union_set_body(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_space *space)
{
	struct isl_obj obj = { isl_obj_set, NULL };
	isl_map *map;

	map = isl_set_universe(space);
	if (isl_stream_eat(s, '{') < 0)
		goto error;
	obj = obj_read_disjuncts(s, v, map);
	if (isl_stream_eat(s, '}') < 0)
		goto error;
	isl_map_free(map);

	return extract_union_set(s->ctx, obj);
error:
	obj.type->free(obj.v);
	isl_map_free(map);
	return NULL;
}

/* Read the body of an isl_multi_union_pw_aff from "s",
 * i.e., everything except the parameter specification.
 * "v" contains a description of the identifiers parsed so far.
 * The parameters, if any, are specified by "space".
 *
 * In particular, handle the special case with shared domain constraints.
 * These are specified as
 *
 *	([...] : ...)
 *
 * and are especially useful for setting the explicit domain
 * of a zero-dimensional isl_multi_union_pw_aff.
 * The core isl_multi_union_pw_aff body ([...]) is read by
 * read_multi_union_pw_aff_body_core.
 */
static __isl_give isl_multi_union_pw_aff *read_multi_union_pw_aff_body(
	__isl_keep isl_stream *s, struct vars *v, __isl_take isl_space *space)
{
	isl_multi_union_pw_aff *mupa;

	if (!isl_stream_next_token_is(s, '('))
		return read_multi_union_pw_aff_body_core(s, v, space);

	if (isl_stream_eat(s, '(') < 0)
		goto error;
	mupa = read_multi_union_pw_aff_body_core(s, v, isl_space_copy(space));
	if (isl_stream_eat_if_available(s, ':')) {
		isl_union_set *dom;

		dom = read_union_set_body(s, v, space);
		mupa = isl_multi_union_pw_aff_intersect_domain(mupa, dom);
	} else {
		isl_space_free(space);
	}
	if (isl_stream_eat(s, ')') < 0)
		return isl_multi_union_pw_aff_free(mupa);

	return mupa;
error:
	isl_space_free(space);
	return NULL;
}

/* Read an isl_multi_union_pw_aff from "s".
 *
 * The input has the form
 *
 *	[{ [..] : ... ; [..] : ... }, { [..] : ... ; [..] : ... }]
 *
 * or
 *
 *	[..] -> [{ [..] : ... ; [..] : ... }, { [..] : ... ; [..] : ... }]
 *
 * Additionally, a shared domain may be specified as
 *
 *	([..] : ...)
 *
 * or
 *
 *	[..] -> ([..] : ...)
 *
 * The first case is handled by the caller, the second case
 * is handled by read_multi_union_pw_aff_body.
 *
 * We first check for the special case of an empty tuple "[]".
 * Then we check if there are any parameters.
 * Finally, read the tuple and construct the result.
 */
static __isl_give isl_multi_union_pw_aff *read_multi_union_pw_aff_core(
	__isl_keep isl_stream *s)
{
	struct vars *v;
	isl_set *dom = NULL;
	isl_space *space;
	isl_multi_union_pw_aff *mupa = NULL;

	if (next_is_empty_tuple(s)) {
		if (isl_stream_eat(s, '['))
			return NULL;
		if (isl_stream_eat(s, ']'))
			return NULL;
		space = isl_space_set_alloc(s->ctx, 0, 0);
		return isl_multi_union_pw_aff_zero(space);
	}

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_param_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	space = isl_set_get_space(dom);
	isl_set_free(dom);
	mupa = read_multi_union_pw_aff_body(s, v, space);

	vars_free(v);

	return mupa;
error:
	vars_free(v);
	isl_set_free(dom);
	isl_multi_union_pw_aff_free(mupa);
	return NULL;
}

/* Read an isl_multi_union_pw_aff from "s".
 *
 * In particular, handle the special case with shared domain constraints.
 * These are specified as
 *
 *	([...] : ...)
 *
 * and are especially useful for setting the explicit domain
 * of a zero-dimensional isl_multi_union_pw_aff.
 * The core isl_multi_union_pw_aff ([...]) is read by
 * read_multi_union_pw_aff_core.
 */
__isl_give isl_multi_union_pw_aff *isl_stream_read_multi_union_pw_aff(
	__isl_keep isl_stream *s)
{
	isl_multi_union_pw_aff *mupa;

	if (!isl_stream_next_token_is(s, '('))
		return read_multi_union_pw_aff_core(s);

	if (isl_stream_eat(s, '(') < 0)
		return NULL;
	mupa = read_multi_union_pw_aff_core(s);
	if (isl_stream_eat_if_available(s, ':')) {
		isl_union_set *dom;

		dom = isl_stream_read_union_set(s);
		mupa = isl_multi_union_pw_aff_intersect_domain(mupa, dom);
	}
	if (isl_stream_eat(s, ')') < 0)
		return isl_multi_union_pw_aff_free(mupa);
	return mupa;
}

/* Read an isl_multi_union_pw_aff from "str".
 */
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_read_from_str(
	isl_ctx *ctx, const char *str)
{
	isl_multi_union_pw_aff *mupa;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	mupa = isl_stream_read_multi_union_pw_aff(s);
	isl_stream_free(s);
	return mupa;
}

__isl_give isl_union_pw_qpolynomial *isl_stream_read_union_pw_qpolynomial(
	__isl_keep isl_stream *s)
{
	struct isl_obj obj;

	obj = obj_read(s);
	if (obj.type == isl_obj_pw_qpolynomial) {
		obj.type = isl_obj_union_pw_qpolynomial;
		obj.v = isl_union_pw_qpolynomial_from_pw_qpolynomial(obj.v);
	}
	if (obj.v)
		isl_assert(s->ctx, obj.type == isl_obj_union_pw_qpolynomial,
			   goto error);

	return obj.v;
error:
	obj.type->free(obj.v);
	return NULL;
}

__isl_give isl_union_pw_qpolynomial *isl_union_pw_qpolynomial_read_from_str(
	isl_ctx *ctx, const char *str)
{
	isl_union_pw_qpolynomial *upwqp;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	upwqp = isl_stream_read_union_pw_qpolynomial(s);
	isl_stream_free(s);
	return upwqp;
}
