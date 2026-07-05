// RUN: llvm-tblgen %s | FileCheck %s

class ToLower<string str> {
  string s = !tolower(str);
}

class ToUpper<string str> {
  string s = !toupper(str);
}

// CHECK-LABEL: def LOWER1 {
// CHECK: string s = "str";
// CHECK: }
def LOWER1: ToLower<"STR">;

// CHECK-LABEL: def LOWER2 {
// CHECK: string s = "str";
// CHECK: }
def LOWER2 : ToLower<"Str">;

// CHECK-LABEL: def LOWER3 {
// CHECK: string s = "str";
// CHECK: }
def LOWER3 : ToLower<"STr">;

// CHECK-LABEL: def UPPER1 {
// CHECK: string s = "STR";
// CHECK: }
def UPPER1 : ToUpper<"str">;

// CHECK-LABEL: def UPPER2 {
// CHECK: string s = "STR";
// CHECK: }
def UPPER2 : ToUpper<"sTr">;

// CHECK-LABEL: def UPPER3 {
// CHECK: string s = "STR";
// CHECK: }
def UPPER3 : ToUpper<"sTR">;