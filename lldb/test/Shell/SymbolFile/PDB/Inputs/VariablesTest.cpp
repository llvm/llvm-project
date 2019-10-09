typedef int IntTypedef;
IntTypedef g_IntVar;  // Testing globals.

typedef enum Enum { // Testing constants.
  RED,
  GREEN,
  BLUE
} EnumTypedef;
EnumTypedef g_EnumVar;  // Testing members.

// FIXME: `sg_IntVar` appears both in global scope's children and compiland's
// children but with different symbol's id.
static int sg_IntVar = -1;  // Testing file statics.

// FIXME: `g_Const` appears both in global scope's children and compiland's
// children but with different symbol's id.
const int g_Const = 0x88;  // Testing constant data.
const int *g_pConst = &g_Const; // Avoid optimizing the const away

thread_local int g_tls = 0;  // Testing thread-local storage.

class Class {
  static int m_StaticClassMember;
public:
  explicit Class(int a) {}
  void Func() {}
};
int Class::m_StaticClassMember = 10; // Testing static class members.
Class ClassVar(1);

int f(int var_arg1, int var_arg2) {  // Testing parameters.
  long same_name_var = -1;
  return 1;
}

int same_name_var = 100;
int main() {
  int same_name_var = 0;  // Testing locals.
  const char local_const = 0x1;

  // FIXME: 'local_CString` is not found through compiland's children.
  const char local_CString[] = "abc";  // Testing constant string.
  const char *local_pCString = local_CString; // Avoid optimizing the const away

  int a = 10;
  a++;

  ClassVar.Func();
  return 0;
}
