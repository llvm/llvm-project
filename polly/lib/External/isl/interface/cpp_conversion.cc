/*
 * Copyright 2018      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#include <stdio.h>
#include <map>
#include <string>

#include "cpp.h"
#include "cpp_conversion.h"

/* If "clazz" describes a subclass of a C type, then print code
 * for converting an object of the class derived from the C type
 * to the subclass.  Do this by first converting this class
 * to the immediate superclass of the subclass and then converting
 * from this superclass to the subclass.
 */
void cpp_conversion_generator::cast(const isl_class &clazz, const char *to)
{
	string name = cpp_generator::type2cpp(clazz);

	if (!clazz.is_type_subclass())
		return;

	cast(classes[clazz.superclass_name], to);
	printf(".as<%s%s>()", to, name.c_str());
}

/* Print a function called "function" for converting objects of
 * "clazz" from the "from" bindings to the "to" bindings.
 * If "clazz" describes a subclass of a C type, then the result
 * of the conversion between bindings is derived from the C type and
 * needs to be converted back to the subclass.
 */
void cpp_conversion_generator::convert(const isl_class &clazz,
	const char *from, const char *to, const char *function)
{
	string name = cpp_generator::type2cpp(clazz);

	printf("%s%s %s(%s%s obj) {\n",
		to, name.c_str(), function, from, name.c_str());
	printf("\t""return %s""manage(obj.copy())", to);
	cast(clazz, to);
	printf(";\n");
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
void cpp_conversion_generator::print(const isl_class &clazz)
{
	convert(clazz, "", "checked::", "check");
	convert(clazz, "checked::", "", "uncheck");
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
