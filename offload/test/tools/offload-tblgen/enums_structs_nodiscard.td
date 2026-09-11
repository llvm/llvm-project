// RUN: %offload-tblgen -gen-api -I %S/../../../liboffload/API %s | %fcheck-generic --check-prefix=CHECK-API

// Check nodiscard support for enums and structs

include "APIDefs.td"

def OL_NODISCARD : Macro {
  let desc = "Marks a function whose result should not be discarded";
  let condition = "(defined(__cplusplus) && __cplusplus >= 201703L) || "
                  "(defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)";
  let value = "[[nodiscard]]";
  let alt_value = "";
}

def my_bitfield_nodiscard_enum_t : Enum {
  let desc = "A bitfield enum marked nodiscard";
  let is_bit_field = 1;
  let nodiscard = 1;
  let etors = [
    Etor<"FLAG_A", "Flag A">,
    Etor<"FLAG_B", "Flag B">
  ];
}

def my_nodiscard_enum_t : Enum {
  let desc = "An enum marked nodiscard";
  let nodiscard = 1;
  let etors = [
    Etor<"ERROR_ONE", "First error">,
    Etor<"ERROR_TWO", "Second error">
  ];
}

def my_regular_enum_t : Enum {
  let desc = "A regular enum without nodiscard";
  let etors = [
    Etor<"VALUE_ONE", "First value">,
    Etor<"VALUE_TWO", "Second value">
  ];
}

def my_typed_nodiscard_enum_t : Enum {
  let desc = "A typed enum marked nodiscard";
  let is_typed = 1;
  let nodiscard = 1;
  let etors = [
    TaggedEtor<"TYPE_A", "uint32_t", "Type A">,
    TaggedEtor<"TYPE_B", "char*", "Type B">
  ];
}

def my_nodiscard_struct_t : Struct {
  let desc = "A struct marked nodiscard";
  let nodiscard = 1;
  let members = [
    StructMember<"int", "errorCode", "Error code">
  ];
}

def my_regular_struct_t : Struct {
  let desc = "A regular struct without nodiscard";
  let members = [
    StructMember<"uint32_t", "field1", "First field">
  ];
}

// CHECK-API: #ifndef OL_NODISCARD
// CHECK-API: #define OL_NODISCARD {{\[\[}}nodiscard{{\]\]}}

// CHECK-API: /// @brief A bitfield enum marked nodiscard
// CHECK-API-NEXT: typedef enum OL_NODISCARD my_bitfield_nodiscard_enum_t {
// CHECK-API: MY_BITFIELD_NODISCARD_ENUM_FLAG_A = 1,
// CHECK-API: MY_BITFIELD_NODISCARD_ENUM_FLAG_B = 2,
// CHECK-API: } my_bitfield_nodiscard_enum_t;

// CHECK-API: /// @brief An enum marked nodiscard
// CHECK-API-NEXT: typedef enum OL_NODISCARD my_nodiscard_enum_t {
// CHECK-API: MY_NODISCARD_ENUM_ERROR_ONE = 0,
// CHECK-API: MY_NODISCARD_ENUM_ERROR_TWO = 1,
// CHECK-API: } my_nodiscard_enum_t;

// CHECK-API: /// @brief A regular enum without nodiscard
// CHECK-API-NEXT: typedef enum my_regular_enum_t {
// CHECK-API: MY_REGULAR_ENUM_VALUE_ONE = 0,
// CHECK-API: } my_regular_enum_t;

// CHECK-API: /// @brief A typed enum marked nodiscard
// CHECK-API-NEXT: typedef enum OL_NODISCARD my_typed_nodiscard_enum_t {
// CHECK-API: [uint32_t] Type A
// CHECK-API: MY_TYPED_NODISCARD_ENUM_TYPE_A = 0,
// CHECK-API: } my_typed_nodiscard_enum_t;

// CHECK-API: /// @brief A struct marked nodiscard
// CHECK-API-NEXT: typedef struct OL_NODISCARD my_nodiscard_struct_t {
// CHECK-API-NEXT: int errorCode;
// CHECK-API: } my_nodiscard_struct_t;

// CHECK-API: /// @brief A regular struct without nodiscard
// CHECK-API-NEXT: typedef struct my_regular_struct_t {
// CHECK-API-NEXT: uint32_t field1;
// CHECK-API: } my_regular_struct_t;
