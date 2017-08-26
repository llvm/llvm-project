#ifndef _SCHEDULE_H
#define _SCHEDULE_H

#include <isl/id.h>
#include <isl/set_type.h>
#include <isl/map_type.h>
#include <isl/union_map_type.h>

#include <pet.h>

/* An access to an outer array element or an iterator.
 * Accesses to iterators have an access relation that maps to an unnamed space.
 * An access may be both read and write.
 * If the access relation is empty, then the output dimension may
 * not be equal to the dimension of the corresponding array.
 */
struct gpu_stmt_access {
	/* Access reads elements */
	int read;
	/* Access writes elements */
	int write;
	/* All writes are definite writes. */
	int exact_write;
	/* The number of index expressions specified in the access. */
	int n_index;

	/* May access relation */
	isl_map *access;
	/* May access relation with as domain a mapping from iteration domain
	 * to a reference identifier.
	 */
	isl_map *tagged_access;
	/* The reference id of the corresponding pet_expr. */
	isl_id *ref_id;

	struct gpu_stmt_access *next;
};

struct gpu_stmt {
	isl_id *id;
	struct pet_stmt *stmt;

	/* Linked list of accesses. */
	struct gpu_stmt_access *accesses;
};

__isl_give isl_map *project_out(__isl_take isl_space *dim,
	int len, int first, int n);
__isl_give isl_map *projection(__isl_take isl_space *dim,
	int src_len, int dst_len);
__isl_give isl_set *parametrization(__isl_take isl_space *space,
	int len, int first, __isl_keep isl_id_list *names);
__isl_give isl_set *extend(__isl_take isl_set *set, int dst_len);
__isl_give isl_union_map *align_range(__isl_take isl_union_map *umap);

#endif
