namespace cpp_namespace {
  struct CppStruct {
    int field = 1111;

    int function() {
      return 2222;
    }
  };

  union CppUnion {
    char field_char;
    short field_short;
    int field_int;
  };

  CppStruct GetCppStruct() {
    return CppStruct();
  }

  CppStruct global;

  CppStruct *GetCppStructPtr() {
    return &global;
  }
}

int global = 3333;

int main()
{
  cpp_namespace::CppStruct cpp_struct = cpp_namespace::GetCppStruct();
  cpp_struct.function();

  cpp_namespace::CppStruct &cpp_struct_ref = cpp_struct;
  cpp_struct_ref.function();

  int field = 4444;

  cpp_namespace::CppUnion cpp_union;
  cpp_union.field_int = 5555;

  int cpp_scalar = 6666;

  cpp_namespace::CppStruct cpp_array[16];

  cpp_namespace::CppStruct *cpp_pointer = cpp_namespace::GetCppStructPtr();

  return 0; // Break here
}
