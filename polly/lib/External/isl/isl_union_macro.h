#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef PART
#define PART CAT(isl_,BASE)
#undef UNION
#define UNION CAT(isl_union_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)
#define xS(TYPE,NAME) struct TYPE ## _ ## NAME
#define S(TYPE,NAME) xS(TYPE,NAME)
