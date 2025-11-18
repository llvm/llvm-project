// RUN: %offload-tblgen -gen-get-info-wrappers -I %S/../../../liboffload/API %s | %fcheck-generic

include "APIDefs.td"

def ol_foo_handle_t : Handle {
}

def ol_foo_info_t : Enum {
  let is_typed = 1;
  let etors = [
    TaggedEtor<"INT", "int", "">,
    TaggedEtor<"STRING", "char[]", "">,
    TaggedEtor<"ARRAY", "int[]", "">,
  ];
}

def olGetFooInfo : Function {
  let params = [
    Param<"ol_foo_handle_t", "Foo", "", PARAM_IN>,
    Param<"ol_foo_info_t", "PropName", "", PARAM_IN>,
    Param<"size_t", "PropSize", "", PARAM_IN>,
    TypeTaggedParam<"void*", "PropValue", "array of bytes holding the info.", PARAM_OUT,
      TypeInfo<"PropName", "PropSize">>
  ];
  let returns = [
    Return<"OL_FOO_INVALID">
  ];
}

// CHECK-LABEL: template <ol_foo_info_t Desc> inline auto get_info(ol_foo_handle_t Foo);
// CHECK-NEXT:  template<> inline auto get_info<OL_FOO_INFO_INT>(ol_foo_handle_t Foo) {
// CHECK-NEXT:    int Result;
// CHECK-NEXT:    if (auto Err = olGetFooInfo(Foo, OL_FOO_INFO_INT, 1, &Result))
// CHECK-NEXT:      return std::variant<int, ol_result_t>{Err};
// CHECK-NEXT:    else
// CHECK-NEXT:      return std::variant<int, ol_result_t>{Result};
// CHECK-NEXT:  }
// CHECK-NEXT:  template<> inline auto get_info<OL_FOO_INFO_STRING>(ol_foo_handle_t Foo) {
// CHECK-NEXT:    std::string Result;
// CHECK-NEXT:    size_t ResultSize = 0;
// CHECK-NEXT:    if (auto Err = olGetFooInfoSize(Foo, OL_FOO_INFO_STRING, &ResultSize))
// CHECK-NEXT:      return std::variant<std::string, ol_result_t>{Err};
// CHECK-NEXT:    ResultSize -= 1;
// CHECK-NEXT:    Result.resize(ResultSize);
// CHECK-NEXT:    if (auto Err = olGetFooInfo(Foo, OL_FOO_INFO_STRING, ResultSize, Result.data()))
// CHECK-NEXT:      return std::variant<std::string, ol_result_t>{Err};
// CHECK-NEXT:    else
// CHECK-NEXT:      return std::variant<std::string, ol_result_t>{Result};
// CHECK-NEXT:  }
// CHECK-NEXT:  template<> inline auto get_info<OL_FOO_INFO_ARRAY>(ol_foo_handle_t Foo) {
// CHECK-NEXT:    std::vector<int> Result;
// CHECK-NEXT:    size_t ResultSize = 0;
// CHECK-NEXT:    if (auto Err = olGetFooInfoSize(Foo, OL_FOO_INFO_ARRAY, &ResultSize))
// CHECK-NEXT:      return std::variant<std::vector<int>, ol_result_t>{Err};
// CHECK-NEXT:    assert(ResultSize % sizeof(int) == 0);
// CHECK-NEXT:    ResultSize /= sizeof(int);
// CHECK-NEXT:    Result.resize(ResultSize);
// CHECK-NEXT:    if (auto Err = olGetFooInfo(Foo, OL_FOO_INFO_ARRAY, ResultSize, Result.data()))
// CHECK-NEXT:      return std::variant<std::vector<int>, ol_result_t>{Err};
// CHECK-NEXT:    else
// CHECK-NEXT:      return std::variant<std::vector<int>, ol_result_t>{Result};
// CHECK-NEXT:  }
