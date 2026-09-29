// Object file built using:
// clang -g -mllvm -generate-dwarf-pubnames -o dwarfdump-pubnames.elf-x86_64 \
//    dwarfdump-pubnames.cc  -c

struct C {
  void member_function();
  static int static_member_function();
  static int static_member_variable;
};

int C::static_member_variable = 0;

void C::member_function() {
  static_member_variable = 0;
}

int C::static_member_function() {
  return static_member_variable;
}

C global_variable;

int global_function() {
  return -1;
}

namespace ns {
  void global_namespace_function() {
    global_variable.member_function();
  }
  int global_namespace_variable = 1;
}
