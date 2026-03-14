#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Print a key-value pair of a YAML mapping to "p",
 * with key "name" and value "val".
 */
static __isl_give isl_printer *FN(print_yaml_field,BASE)(
	__isl_take isl_printer *p, const char *name, __isl_keep TYPE *val)
{
	p = isl_printer_print_str(p, name);
	p = isl_printer_yaml_next(p);
	p = isl_printer_print_str(p, "\"");
	p = FN(isl_printer_print,BASE)(p, val);
	p = isl_printer_print_str(p, "\"");
	p = isl_printer_yaml_next(p);

	return p;
}
