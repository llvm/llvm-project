/*
 * Copyright 2018      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#include "cpp.h"
#include "cpp_conversion.h"

/* Print a function called "function" for converting objects of
 * type "name" from the "from" bindings to the "to" bindings.
 */
static void convert(const char *name, const char *from, const char *to,
	const char *function)
{
	printf("%s%s %s(%s%s obj) {\n", to, name, function, from, name);
	printf("\t""return %s""manage(obj.copy());\n", to);
	printf("}\n");
	printf("\n");
}

/* Print functions for converting objects of "clazz"
 * between the default and the checked C++ bindings.
 *
 * The conversion from default to checked is called "check".
 * The inverse conversion is called "uncheck".
 * For example, to "set", the following two functions are generated:
 *
 *	checked::set check(set obj) {
 *		return checked::manage(obj.copy());
 *	}
 *
 *	set uncheck(checked::set obj) {
 *		return manage(obj.copy());
 *	}
 */
static void print(const isl_class &clazz)
{
	string name = cpp_generator::type2cpp(clazz.name);

	convert(name.c_str(), "", "checked::", "check");
	convert(name.c_str(), "checked::", "", "uncheck");
}

/* Generate conversion functions for converting objects between
 * the default and the checked C++ bindings.
 * Do this for each exported class.
 */
void cpp_conversion_generator::generate()
{
	map<string, isl_class>::iterator ci;

	printf("namespace isl {\n\n");
	for (ci = classes.begin(); ci != classes.end(); ++ci)
		print(ci->second);
	printf("} // namespace isl\n");
}
