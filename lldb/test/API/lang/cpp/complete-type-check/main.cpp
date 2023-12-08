struct EmptyClass {};
struct DefinedClass {
  int i;
};
typedef DefinedClass DefinedClassTypedef;

struct FwdClass;
typedef FwdClass FwdClassTypedef;

template <typename T> struct DefinedTemplateClass {};
template <> struct DefinedTemplateClass<int> {};

template <typename T> struct FwdTemplateClass;
template <> struct FwdTemplateClass<int>;

enum class EnumClassFwd;

enum DefinedEnum { Case1 };
enum DefinedEnumClass { Case2 };

EmptyClass empty_class;
DefinedClass defined_class;
DefinedClassTypedef defined_class_typedef;

FwdClass *fwd_class;
FwdClassTypedef *fwd_class_typedef;

DefinedTemplateClass<int> defined_template_class;
FwdTemplateClass<int> *fwd_template_class;

EnumClassFwd *fwd_enum_class = nullptr;

DefinedEnum defined_enum = Case1;
DefinedEnumClass defined_enum_class = DefinedEnumClass::Case2;

int main() {}
