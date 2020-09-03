#define NO_LOC
#ifdef HAS_TYPE
#define OPT_TYPE_PARAM			, enum isl_fold type
#define OPT_TYPE_PARAM_FIRST		enum isl_fold type,
#define OPT_TYPE_ARG(loc)		, loc type
#define OPT_TYPE_ARG_FIRST(loc)		loc type,
#define OPT_SET_TYPE(loc,val)		loc type = (val);
#define OPT_EQUAL_TYPES(loc1, loc2)	((loc1 type) == (loc2 type))
#else
#define OPT_TYPE_PARAM
#define OPT_TYPE_PARAM_FIRST
#define OPT_TYPE_ARG(loc)
#define OPT_TYPE_ARG_FIRST(loc)
#define OPT_SET_TYPE(loc,val)
#define OPT_EQUAL_TYPES(loc1, loc2)	1
#endif
