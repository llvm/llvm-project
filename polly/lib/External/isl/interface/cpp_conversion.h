#include "generator.h"

class cpp_conversion_generator : public generator {
public:
	cpp_conversion_generator(set<RecordDecl *> &exported_types,
		set<FunctionDecl *> exported_functions,
		set<FunctionDecl *> functions) :
		generator(exported_types, exported_functions, functions) {}
	virtual void generate();
};
