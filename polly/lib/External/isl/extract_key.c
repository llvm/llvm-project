/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <string.h>

/* Extract a mapping key from the token "tok".
 * Return KEY_ERROR on error, i.e., if "tok" does not
 * correspond to any known key.
 */
static KEY KEY_EXTRACT(__isl_keep isl_stream *s, struct isl_token *tok)
{
	isl_bool has_string;
	char *name;
	KEY key;
	isl_ctx *ctx;

	has_string = isl_token_has_str(tok);
	if (has_string < 0)
		return KEY_ERROR;
	if (!has_string) {
		isl_stream_error(s, tok, "expecting key");
		return KEY_ERROR;
	}

	ctx = isl_stream_get_ctx(s);
	name = isl_token_get_str(ctx, tok);
	if (!name)
		return KEY_ERROR;

	for (key = 0; key < KEY_END; ++key) {
		if (KEY_STR[key] && !strcmp(name, KEY_STR[key]))
			break;
	}
	free(name);

	if (key >= KEY_END)
		isl_die(ctx, isl_error_invalid, "unknown key",
			return KEY_ERROR);
	return key;
}

/* Read a key from "s" and return the corresponding enum.
 * Return KEY_ERROR on error, i.e., if the first token
 * on the stream does not correspond to any known key.
 */
static KEY KEY_GET(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	KEY key;

	tok = isl_stream_next_token(s);
	key = KEY_EXTRACT(s, tok);
	isl_token_free(tok);

	return key;
}
