#include <isl/ctx.h>
#include <isl_blk.h>

/* "error" stores the last error that has occurred.
 * It is reset to isl_error_none by isl_ctx_reset_error.
 * "error_msg" stores the error message of the last error,
 * while "error_file" and "error_line" specify where the last error occurred.
 * "error_msg" and "error_file" always point to statically allocated
 * strings (if not NULL).
 */
struct isl_ctx {
	int			ref;

	struct isl_stats	*stats;

	int			 opt_allocated;
	struct isl_options	*opt;
	void			*user_opt;
	struct isl_args		*user_args;

	isl_int			zero;
	isl_int			one;
	isl_int			two;
	isl_int			negone;

	isl_int			normalize_gcd;

	int			n_cached;
	int			n_miss;
	struct isl_blk		cache[ISL_BLK_CACHE_SIZE];
	struct isl_hash_table	id_table;

	enum isl_error		error;
	const char		*error_msg;
	const char		*error_file;
	int			error_line;

	int			abort;

	unsigned long		operations;
	unsigned long		max_operations;
};

int isl_ctx_next_operation(isl_ctx *ctx);
