#include "generator.h"

class cpp_conversion_generator : public generator {
	void cast(const isl_class &clazz, const char *to);
	void convert(const isl_class &clazz, const char *from, const char *to,
		const char *function);
	void print(const isl_class &clazz);
public:
	cpp_conversion_generator(SourceManager &SM,
		set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions) :
		generator(SM, exported_types, exported_functions, functions) {}
	virtual void generate();
};
