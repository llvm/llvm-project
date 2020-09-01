#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef EL
#define EL CAT(isl_,BASE)
#undef PW_BASE
#define PW_BASE CAT(pw_,BASE)
#undef PW
#define PW CAT(isl_,PW_BASE)
#undef UNION_BASE
#define UNION_BASE CAT(union_,PW_BASE)
#undef UNION
#define UNION CAT(isl_,UNION_BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Create a union piecewise expression
 * with the given base expression on a universe domain.
 */
__isl_give UNION *FN(FN(UNION,from),BASE)(__isl_take EL *el)
{
	return FN(FN(UNION,from),PW_BASE)(FN(FN(PW,from),BASE)(el));
}
