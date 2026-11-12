#include <isl_printer_private.h>

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

#ifndef PRINT_DUMP_DEFAULT
#define PRINT_DUMP_DEFAULT	1
#endif

void FN(TYPE,dump)(__isl_keep TYPE *obj)
{
	isl_printer *p;

	if (!obj)
		return;
	p = isl_printer_to_file(FN(TYPE,get_ctx)(obj), stderr);
	p = isl_printer_set_dump(p, PRINT_DUMP_DEFAULT);
	p = FN(isl_printer_print,BASE)(p, obj);
	p = isl_printer_end_line(p);
	isl_printer_free(p);
}

#undef PRINT_DUMP_DEFAULT

__isl_give char *FN(TYPE,to_str)(__isl_keep TYPE *obj)
{
	isl_printer *p;
	char *s;

	if (!obj)
		return NULL;
	p = isl_printer_to_str(FN(TYPE,get_ctx)(obj));
	p = FN(isl_printer_print,BASE)(p, obj);
	s = isl_printer_get_str(p);
	isl_printer_free(p);

	return s;
}
