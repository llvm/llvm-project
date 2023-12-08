// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Type<string name, int length, int width> {
  string Name = name;
  int Length = length;
  int Width = width;
}

multiclass OT1<string ss, int l, int w> {
  def _#NAME# : Type<ss, l, w>;
}
multiclass OT2<string ss, int w> {
  defm  v1#NAME# : OT1<!strconcat( "v1", ss),  1, w>;
  defm  v2#NAME# : OT1<!strconcat( "v2", ss),  2, w>;
  defm  v3#NAME# : OT1<!strconcat( "v3", ss),  3, w>;
  defm  v4#NAME# : OT1<!strconcat( "v4", ss),  4, w>;
  defm  v8#NAME# : OT1<!strconcat( "v8", ss),  8, w>;
  defm v16#NAME# : OT1<!strconcat("v16", ss), 16, w>;
}

defm i8 : OT2<"i8", 8>;

multiclass OT3<string ss, int w> {
  defm v32#NAME : OT1<!strconcat("v32", ss), 32, w>;
}

multiclass OT4<string ss, int w> {
  defm v64#NAME : OT1<!strconcat("v64", ss), 64, w>;
}

multiclass OT5<string ss, int w> {
  defm NAME : OT3<ss, w>;
  defm NAME : OT4<ss, w>;
}

defm i16 : OT5<"i16", 16>;

// CHECK: _v16i8
// CHECK: Length = 16
// CHECK: Width = 8

// CHECK: _v1i8
// CHECK: Length = 1
// CHECK: Width = 8

// CHECK: _v2i8
// CHECK: Length = 2
// CHECK: Width = 8

// CHECK: def _v32i16
// CHECK: Length = 32
// CHECK: Width = 16

// CHECK: _v3i8
// CHECK: Length = 3
// CHECK: Width = 8

// CHECK: _v4i8
// CHECK: Length = 4
// CHECK: Width = 8

// CHECK: _v64i16
// CHECK: Length = 64
// CHECK: Width = 16

// CHECK: _v8i8
// CHECK: Length = 8
// CHECK: Width = 8

