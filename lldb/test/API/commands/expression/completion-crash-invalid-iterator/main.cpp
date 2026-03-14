class LoadedByParamClass {};
struct ParamClass {
  LoadedByParamClass some_func();
};
struct SomeClass {
  // LLDB stops in the constructor and then requests
  // possible expression completions. This will iterate over the
  // declarations in the translation unit.
  // The unnamed ParamClass parameter causes that LLDB will add
  // an incomplete ParamClass decl to the translation unit which
  // the code completion will find. Upon inspecting the ParamClass
  // decl to see if it can be used to provide any useful completions,
  // Clang will complete it and load all its members.
  // This causes that its member function some_func is loaded which in turn
  // loads the LoadedByParamClass decl. When LoadedByParamClass
  // is created it will be added to the translation unit which
  // will invalidate all iterators that currently iterate over
  // the translation unit. The iterator we use for code completion
  // is now invalidated and LLDB crashes.
  SomeClass(ParamClass) {}
};
int main() { ParamClass e; SomeClass y(e); }
