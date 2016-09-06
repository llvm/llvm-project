; "PLAIN" - No optimizations. This tests the default target layout
; constant folder.
; RUN: opt -S -o - < %s | FileCheck --check-prefix=PLAIN %s

; "OPT" - Optimizations but no targetdata. This tests default target layout
; folding in the optimizers.
; RUN: opt -S -o - -instcombine -globalopt < %s | FileCheck --check-prefix=OPT %s

; "TO" - Optimizations and targetdata. This tests target-dependent
; folding in the optimizers.
; RUN: opt -S -o - -instcombine -globalopt -default-data-layout="e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64" < %s | FileCheck --check-prefix=TO %s

; "SCEV" - ScalarEvolution with default target layout
; RUN: opt -analyze -scalar-evolution < %s | FileCheck --check-prefix=SCEV %s


; The automatic constant folder in opt does not have targetdata access, so
; it can't fold gep arithmetic, in general. However, the constant folder run
; from instcombine and global opt can use targetdata.

; PLAIN: @G8 = global i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -1)
; PLAIN: @G1 = global i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -1)
; PLAIN: @F8 = global i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -2)
; PLAIN: @F1 = global i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -2)
; PLAIN: @H8 = global i8* getelementptr (i8, i8* null, i32 -1)
; PLAIN: @H1 = global i1* getelementptr (i1, i1* null, i32 -1)
; OPT: @G8 = local_unnamed_addr global i8* null
; OPT: @G1 = local_unnamed_addr global i1* null
; OPT: @F8 = local_unnamed_addr global i8* inttoptr (i64 -1 to i8*)
; OPT: @F1 = local_unnamed_addr global i1* inttoptr (i64 -1 to i1*)
; OPT: @H8 = local_unnamed_addr global i8* inttoptr (i64 -1 to i8*)
; OPT: @H1 = local_unnamed_addr global i1* inttoptr (i64 -1 to i1*)
; TO: @G8 = local_unnamed_addr global i8* null
; TO: @G1 = local_unnamed_addr global i1* null
; TO: @F8 = local_unnamed_addr global i8* inttoptr (i64 -1 to i8*)
; TO: @F1 = local_unnamed_addr global i1* inttoptr (i64 -1 to i1*)
; TO: @H8 = local_unnamed_addr global i8* inttoptr (i64 -1 to i8*)
; TO: @H1 = local_unnamed_addr global i1* inttoptr (i64 -1 to i1*)

@G8 = global i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -1)
@G1 = global i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -1)
@F8 = global i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -2)
@F1 = global i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -2)
@H8 = global i8* getelementptr (i8, i8* inttoptr (i32 0 to i8*), i32 -1)
@H1 = global i1* getelementptr (i1, i1* inttoptr (i32 0 to i1*), i32 -1)

; The target-independent folder should be able to do some clever
; simplifications on sizeof, alignof, and offsetof expressions. The
; target-dependent folder should fold these down to constants.

; PLAIN: @a = constant i64 mul (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 2310)
; PLAIN: @b = constant i64 ptrtoint (double* getelementptr ({ i1, double }, { i1, double }* null, i64 0, i32 1) to i64)
; PLAIN: @c = constant i64 mul nuw (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 2)
; PLAIN: @d = constant i64 mul nuw (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 11)
; PLAIN: @e = constant i64 ptrtoint (double* getelementptr ({ double, float, double, double }, { double, float, double, double }* null, i64 0, i32 2) to i64)
; PLAIN: @f = constant i64 1
; PLAIN: @g = constant i64 ptrtoint (double* getelementptr ({ i1, double }, { i1, double }* null, i64 0, i32 1) to i64)
; PLAIN: @h = constant i64 ptrtoint (i1** getelementptr (i1*, i1** null, i32 1) to i64)
; PLAIN: @i = constant i64 ptrtoint (i1** getelementptr ({ i1, i1* }, { i1, i1* }* null, i64 0, i32 1) to i64)
; OPT: @a = local_unnamed_addr constant i64 18480
; OPT: @b = local_unnamed_addr constant i64 8
; OPT: @c = local_unnamed_addr constant i64 16
; OPT: @d = local_unnamed_addr constant i64 88
; OPT: @e = local_unnamed_addr constant i64 16
; OPT: @f = local_unnamed_addr constant i64 1
; OPT: @g = local_unnamed_addr constant i64 8
; OPT: @h = local_unnamed_addr constant i64 8
; OPT: @i = local_unnamed_addr constant i64 8
; TO: @a = local_unnamed_addr constant i64 18480
; TO: @b = local_unnamed_addr constant i64 8
; TO: @c = local_unnamed_addr constant i64 16
; TO: @d = local_unnamed_addr constant i64 88
; TO: @e = local_unnamed_addr constant i64 16
; TO: @f = local_unnamed_addr constant i64 1
; TO: @g = local_unnamed_addr constant i64 8
; TO: @h = local_unnamed_addr constant i64 8
; TO: @i = local_unnamed_addr constant i64 8

@a = constant i64 mul (i64 3, i64 mul (i64 ptrtoint ({[7 x double], [7 x double]}* getelementptr ({[7 x double], [7 x double]}, {[7 x double], [7 x double]}* null, i64 11) to i64), i64 5))
@b = constant i64 ptrtoint ([13 x double]* getelementptr ({i1, [13 x double]}, {i1, [13 x double]}* null, i64 0, i32 1) to i64)
@c = constant i64 ptrtoint (double* getelementptr ({double, double, double, double}, {double, double, double, double}* null, i64 0, i32 2) to i64)
@d = constant i64 ptrtoint (double* getelementptr ([13 x double], [13 x double]* null, i64 0, i32 11) to i64)
@e = constant i64 ptrtoint (double* getelementptr ({double, float, double, double}, {double, float, double, double}* null, i64 0, i32 2) to i64)
@f = constant i64 ptrtoint (<{ i16, i128 }>* getelementptr ({i1, <{ i16, i128 }>}, {i1, <{ i16, i128 }>}* null, i64 0, i32 1) to i64)
@g = constant i64 ptrtoint ({double, double}* getelementptr ({i1, {double, double}}, {i1, {double, double}}* null, i64 0, i32 1) to i64)
@h = constant i64 ptrtoint (double** getelementptr (double*, double** null, i64 1) to i64)
@i = constant i64 ptrtoint (double** getelementptr ({i1, double*}, {i1, double*}* null, i64 0, i32 1) to i64)

; The target-dependent folder should cast GEP indices to integer-sized pointers.

; PLAIN: @M = constant i64* getelementptr (i64, i64* null, i32 1)
; PLAIN: @N = constant i64* getelementptr ({ i64, i64 }, { i64, i64 }* null, i32 0, i32 1)
; PLAIN: @O = constant i64* getelementptr ([2 x i64], [2 x i64]* null, i32 0, i32 1)
; OPT: @M = local_unnamed_addr constant i64* inttoptr (i64 8 to i64*)
; OPT: @N = local_unnamed_addr constant i64* inttoptr (i64 8 to i64*)
; OPT: @O = local_unnamed_addr constant i64* inttoptr (i64 8 to i64*)
; TO: @M = local_unnamed_addr constant i64* inttoptr (i64 8 to i64*)
; TO: @N = local_unnamed_addr constant i64* inttoptr (i64 8 to i64*)
; TO: @O = local_unnamed_addr constant i64* inttoptr (i64 8 to i64*)

@M = constant i64* getelementptr (i64, i64* null, i32 1)
@N = constant i64* getelementptr ({ i64, i64 }, { i64, i64 }* null, i32 0, i32 1)
@O = constant i64* getelementptr ([2 x i64], [2 x i64]* null, i32 0, i32 1)

; Fold GEP of a GEP. Very simple cases are folded without targetdata.

; PLAIN: @Y = global [3 x { i32, i32 }]* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 2)
; PLAIN: @Z = global i32* getelementptr inbounds (i32, i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1)
; OPT: @Y = local_unnamed_addr global [3 x { i32, i32 }]* getelementptr ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 2)
; OPT: @Z = local_unnamed_addr global i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 1)
; TO: @Y = local_unnamed_addr global [3 x { i32, i32 }]* getelementptr ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 2)
; TO: @Z = local_unnamed_addr global i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 1)

@ext = external global [3 x { i32, i32 }]
@Y = global [3 x { i32, i32 }]* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 1), i64 1)
@Z = global i32* getelementptr inbounds (i32, i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1)

; Duplicate all of the above as function return values rather than
; global initializers.

; PLAIN: define i8* @goo8() #0 {
; PLAIN:   %t = bitcast i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -1) to i8*
; PLAIN:   ret i8* %t
; PLAIN: }
; PLAIN: define i1* @goo1() #0 {
; PLAIN:   %t = bitcast i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -1) to i1*
; PLAIN:   ret i1* %t
; PLAIN: }
; PLAIN: define i8* @foo8() #0 {
; PLAIN:   %t = bitcast i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -2) to i8*
; PLAIN:   ret i8* %t
; PLAIN: }
; PLAIN: define i1* @foo1() #0 {
; PLAIN:   %t = bitcast i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -2) to i1*
; PLAIN:   ret i1* %t
; PLAIN: }
; PLAIN: define i8* @hoo8() #0 {
; PLAIN:   %t = bitcast i8* getelementptr (i8, i8* null, i32 -1) to i8*
; PLAIN:   ret i8* %t
; PLAIN: }
; PLAIN: define i1* @hoo1() #0 {
; PLAIN:   %t = bitcast i1* getelementptr (i1, i1* null, i32 -1) to i1*
; PLAIN:   ret i1* %t
; PLAIN: }
; OPT: define i8* @goo8() local_unnamed_addr #0 {
; OPT:   ret i8* null
; OPT: }
; OPT: define i1* @goo1() local_unnamed_addr #0 {
; OPT:   ret i1* null
; OPT: }
; OPT: define i8* @foo8() local_unnamed_addr #0 {
; OPT:   ret i8* inttoptr (i64 -1 to i8*)
; OPT: }
; OPT: define i1* @foo1() local_unnamed_addr #0 {
; OPT:   ret i1* inttoptr (i64 -1 to i1*)
; OPT: }
; OPT: define i8* @hoo8() local_unnamed_addr #0 {
; OPT:   ret i8* inttoptr (i64 -1 to i8*)
; OPT: }
; OPT: define i1* @hoo1() local_unnamed_addr #0 {
; OPT:   ret i1* inttoptr (i64 -1 to i1*)
; OPT: }
; TO: define i8* @goo8() local_unnamed_addr #0 {
; TO:   ret i8* null
; TO: }
; TO: define i1* @goo1() local_unnamed_addr #0 {
; TO:   ret i1* null
; TO: }
; TO: define i8* @foo8() local_unnamed_addr #0 {
; TO:   ret i8* inttoptr (i64 -1 to i8*)
; TO: }
; TO: define i1* @foo1() local_unnamed_addr #0 {
; TO:   ret i1* inttoptr (i64 -1 to i1*)
; TO: }
; TO: define i8* @hoo8() local_unnamed_addr #0 {
; TO:   ret i8* inttoptr (i64 -1 to i8*)
; TO: }
; TO: define i1* @hoo1() local_unnamed_addr #0 {
; TO:   ret i1* inttoptr (i64 -1 to i1*)
; TO: }
; SCEV: Classifying expressions for: @goo8
; SCEV:   %t = bitcast i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -1) to i8*
; SCEV:   -->  (-1 + inttoptr (i32 1 to i8*))
; SCEV: Classifying expressions for: @goo1
; SCEV:   %t = bitcast i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -1) to i1*
; SCEV:   -->  (-1 + inttoptr (i32 1 to i1*))
; SCEV: Classifying expressions for: @foo8
; SCEV:   %t = bitcast i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -2) to i8*
; SCEV:   -->  (-2 + inttoptr (i32 1 to i8*))
; SCEV: Classifying expressions for: @foo1
; SCEV:   %t = bitcast i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -2) to i1*
; SCEV:   -->  (-2 + inttoptr (i32 1 to i1*))
; SCEV: Classifying expressions for: @hoo8
; SCEV:   -->  -1
; SCEV: Classifying expressions for: @hoo1
; SCEV:   -->  -1

define i8* @goo8() nounwind {
  %t = bitcast i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -1) to i8*
  ret i8* %t
}
define i1* @goo1() nounwind {
  %t = bitcast i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -1) to i1*
  ret i1* %t
}
define i8* @foo8() nounwind {
  %t = bitcast i8* getelementptr (i8, i8* inttoptr (i32 1 to i8*), i32 -2) to i8*
  ret i8* %t
}
define i1* @foo1() nounwind {
  %t = bitcast i1* getelementptr (i1, i1* inttoptr (i32 1 to i1*), i32 -2) to i1*
  ret i1* %t
}
define i8* @hoo8() nounwind {
  %t = bitcast i8* getelementptr (i8, i8* inttoptr (i32 0 to i8*), i32 -1) to i8*
  ret i8* %t
}
define i1* @hoo1() nounwind {
  %t = bitcast i1* getelementptr (i1, i1* inttoptr (i32 0 to i1*), i32 -1) to i1*
  ret i1* %t
}

; PLAIN: define i64 @fa() #0 {
; PLAIN:   %t = bitcast i64 mul (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 2310) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fb() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (double* getelementptr ({ i1, double }, { i1, double }* null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fc() #0 {
; PLAIN:   %t = bitcast i64 mul nuw (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 2) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fd() #0 {
; PLAIN:   %t = bitcast i64 mul nuw (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 11) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fe() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (double* getelementptr ({ double, float, double, double }, { double, float, double, double }* null, i64 0, i32 2) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @ff() #0 {
; PLAIN:   %t = bitcast i64 1 to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fg() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (double* getelementptr ({ i1, double }, { i1, double }* null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fh() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (i1** getelementptr (i1*, i1** null, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fi() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (i1** getelementptr ({ i1, i1* }, { i1, i1* }* null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; OPT: define i64 @fa() local_unnamed_addr #0 {
; OPT:   ret i64 18480
; OPT: }
; OPT: define i64 @fb() local_unnamed_addr #0 {
; OPT:   ret i64 8
; OPT: }
; OPT: define i64 @fc() local_unnamed_addr #0 {
; OPT:   ret i64 16
; OPT: }
; OPT: define i64 @fd() local_unnamed_addr #0 {
; OPT:   ret i64 88
; OPT: }
; OPT: define i64 @fe() local_unnamed_addr #0 {
; OPT:   ret i64 16
; OPT: }
; OPT: define i64 @ff() local_unnamed_addr #0 {
; OPT:   ret i64 1
; OPT: }
; OPT: define i64 @fg() local_unnamed_addr #0 {
; OPT:   ret i64 8
; OPT: }
; OPT: define i64 @fh() local_unnamed_addr #0 {
; OPT:   ret i64 8
; OPT: }
; OPT: define i64 @fi() local_unnamed_addr #0 {
; OPT:   ret i64 8
; OPT: }
; TO: define i64 @fa() local_unnamed_addr #0 {
; TO:   ret i64 18480
; TO: }
; TO: define i64 @fb() local_unnamed_addr #0 {
; TO:   ret i64 8
; TO: }
; TO: define i64 @fc() local_unnamed_addr #0 {
; TO:   ret i64 16
; TO: }
; TO: define i64 @fd() local_unnamed_addr #0 {
; TO:   ret i64 88
; TO: }
; TO: define i64 @fe() local_unnamed_addr #0 {
; TO:   ret i64 16
; TO: }
; TO: define i64 @ff() local_unnamed_addr #0 {
; TO:   ret i64 1
; TO: }
; TO: define i64 @fg() local_unnamed_addr #0 {
; TO:   ret i64 8
; TO: }
; TO: define i64 @fh() local_unnamed_addr #0 {
; TO:   ret i64 8
; TO: }
; TO: define i64 @fi() local_unnamed_addr #0 {
; TO:   ret i64 8
; TO: }
; SCEV: Classifying expressions for: @fa
; SCEV:   %t = bitcast i64 mul (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 2310) to i64
; SCEV:   -->  (2310 * sizeof(double))
; SCEV: Classifying expressions for: @fb
; SCEV:   %t = bitcast i64 ptrtoint (double* getelementptr ({ i1, double }, { i1, double }* null, i64 0, i32 1) to i64) to i64
; SCEV:   -->  alignof(double)
; SCEV: Classifying expressions for: @fc
; SCEV:   %t = bitcast i64 mul nuw (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 2) to i64
; SCEV:   -->  (2 * sizeof(double))
; SCEV: Classifying expressions for: @fd
; SCEV:   %t = bitcast i64 mul nuw (i64 ptrtoint (double* getelementptr (double, double* null, i32 1) to i64), i64 11) to i64
; SCEV:   -->  (11 * sizeof(double))
; SCEV: Classifying expressions for: @fe
; SCEV:   %t = bitcast i64 ptrtoint (double* getelementptr ({ double, float, double, double }, { double, float, double, double }* null, i64 0, i32 2) to i64) to i64
; SCEV:   -->  offsetof({ double, float, double, double }, 2)
; SCEV: Classifying expressions for: @ff
; SCEV:   %t = bitcast i64 1 to i64
; SCEV:   -->  1
; SCEV: Classifying expressions for: @fg
; SCEV:   %t = bitcast i64 ptrtoint (double* getelementptr ({ i1, double }, { i1, double }* null, i64 0, i32 1) to i64) to i64
; SCEV:   -->  alignof(double)
; SCEV: Classifying expressions for: @fh
; SCEV:   %t = bitcast i64 ptrtoint (i1** getelementptr (i1*, i1** null, i32 1) to i64) to i64
; SCEV:   -->  sizeof(i1*)
; SCEV: Classifying expressions for: @fi
; SCEV:   %t = bitcast i64 ptrtoint (i1** getelementptr ({ i1, i1* }, { i1, i1* }* null, i64 0, i32 1) to i64) to i64
; SCEV:   -->  alignof(i1*)

define i64 @fa() nounwind {
  %t = bitcast i64 mul (i64 3, i64 mul (i64 ptrtoint ({[7 x double], [7 x double]}* getelementptr ({[7 x double], [7 x double]}, {[7 x double], [7 x double]}* null, i64 11) to i64), i64 5)) to i64
  ret i64 %t
}
define i64 @fb() nounwind {
  %t = bitcast i64 ptrtoint ([13 x double]* getelementptr ({i1, [13 x double]}, {i1, [13 x double]}* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fc() nounwind {
  %t = bitcast i64 ptrtoint (double* getelementptr ({double, double, double, double}, {double, double, double, double}* null, i64 0, i32 2) to i64) to i64
  ret i64 %t
}
define i64 @fd() nounwind {
  %t = bitcast i64 ptrtoint (double* getelementptr ([13 x double], [13 x double]* null, i64 0, i32 11) to i64) to i64
  ret i64 %t
}
define i64 @fe() nounwind {
  %t = bitcast i64 ptrtoint (double* getelementptr ({double, float, double, double}, {double, float, double, double}* null, i64 0, i32 2) to i64) to i64
  ret i64 %t
}
define i64 @ff() nounwind {
  %t = bitcast i64 ptrtoint (<{ i16, i128 }>* getelementptr ({i1, <{ i16, i128 }>}, {i1, <{ i16, i128 }>}* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fg() nounwind {
  %t = bitcast i64 ptrtoint ({double, double}* getelementptr ({i1, {double, double}}, {i1, {double, double}}* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fh() nounwind {
  %t = bitcast i64 ptrtoint (double** getelementptr (double*, double** null, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fi() nounwind {
  %t = bitcast i64 ptrtoint (double** getelementptr ({i1, double*}, {i1, double*}* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}

; PLAIN: define i64* @fM() #0 {
; PLAIN:   %t = bitcast i64* getelementptr (i64, i64* null, i32 1) to i64*
; PLAIN:   ret i64* %t
; PLAIN: }
; PLAIN: define i64* @fN() #0 {
; PLAIN:   %t = bitcast i64* getelementptr ({ i64, i64 }, { i64, i64 }* null, i32 0, i32 1) to i64*
; PLAIN:   ret i64* %t
; PLAIN: }
; PLAIN: define i64* @fO() #0 {
; PLAIN:   %t = bitcast i64* getelementptr ([2 x i64], [2 x i64]* null, i32 0, i32 1) to i64*
; PLAIN:   ret i64* %t
; PLAIN: }
; OPT: define i64* @fM() local_unnamed_addr #0 {
; OPT:   ret i64* inttoptr (i64 8 to i64*)
; OPT: }
; OPT: define i64* @fN() local_unnamed_addr #0 {
; OPT:   ret i64* inttoptr (i64 8 to i64*)
; OPT: }
; OPT: define i64* @fO() local_unnamed_addr #0 {
; OPT:   ret i64* inttoptr (i64 8 to i64*)
; OPT: }
; TO: define i64* @fM() local_unnamed_addr #0 {
; TO:   ret i64* inttoptr (i64 8 to i64*)
; TO: }
; TO: define i64* @fN() local_unnamed_addr #0 {
; TO:   ret i64* inttoptr (i64 8 to i64*)
; TO: }
; TO: define i64* @fO() local_unnamed_addr #0 {
; TO:   ret i64* inttoptr (i64 8 to i64*)
; TO: }
; SCEV: Classifying expressions for: @fM
; SCEV:   %t = bitcast i64* getelementptr (i64, i64* null, i32 1) to i64*
; SCEV:   -->  8
; SCEV: Classifying expressions for: @fN
; SCEV:   %t = bitcast i64* getelementptr ({ i64, i64 }, { i64, i64 }* null, i32 0, i32 1) to i64*
; SCEV:   -->  8
; SCEV: Classifying expressions for: @fO
; SCEV:   %t = bitcast i64* getelementptr ([2 x i64], [2 x i64]* null, i32 0, i32 1) to i64*
; SCEV:   -->  8

define i64* @fM() nounwind {
  %t = bitcast i64* getelementptr (i64, i64* null, i32 1) to i64*
  ret i64* %t
}
define i64* @fN() nounwind {
  %t = bitcast i64* getelementptr ({ i64, i64 }, { i64, i64 }* null, i32 0, i32 1) to i64*
  ret i64* %t
}
define i64* @fO() nounwind {
  %t = bitcast i64* getelementptr ([2 x i64], [2 x i64]* null, i32 0, i32 1) to i64*
  ret i64* %t
}

; PLAIN: define i32* @fZ() #0 {
; PLAIN:   %t = bitcast i32* getelementptr inbounds (i32, i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1) to i32*
; PLAIN:   ret i32* %t
; PLAIN: }
; OPT: define i32* @fZ() local_unnamed_addr #0 {
; OPT:   ret i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 1)
; OPT: }
; TO: define i32* @fZ() local_unnamed_addr #0 {
; TO:   ret i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 1)
; TO: }
; SCEV: Classifying expressions for: @fZ
; SCEV:   %t = bitcast i32* getelementptr inbounds (i32, i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1) to i32*
; SCEV:   -->  (12 + @ext)

define i32* @fZ() nounwind {
  %t = bitcast i32* getelementptr inbounds (i32, i32* getelementptr inbounds ([3 x { i32, i32 }], [3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1) to i32*
  ret i32* %t
}

; PR15262 - Check GEP folding with casts between address spaces.

@p0 = global [4 x i8] zeroinitializer, align 1
@p12 = addrspace(12) global [4 x i8] zeroinitializer, align 1

define i8* @different_addrspace() nounwind noinline {
; OPT: different_addrspace
  %p = getelementptr inbounds i8, i8* addrspacecast ([4 x i8] addrspace(12)* @p12 to i8*),
                                  i32 2
  ret i8* %p
; OPT: ret i8* getelementptr ([4 x i8], [4 x i8]* addrspacecast ([4 x i8] addrspace(12)* @p12 to [4 x i8]*), i64 0, i64 2)
}

define i8* @same_addrspace() nounwind noinline {
; OPT: same_addrspace
  %p = getelementptr inbounds i8, i8* bitcast ([4 x i8] * @p0 to i8*), i32 2
  ret i8* %p
; OPT: ret i8* getelementptr inbounds ([4 x i8], [4 x i8]* @p0, i64 0, i64 2)
}

@gv1 = internal global i32 1
@gv2 = internal global [1 x i32] [ i32 2 ]
@gv3 = internal global [1 x i32] [ i32 2 ]

; Handled by TI-independent constant folder
define i1 @gv_gep_vs_gv() {
  ret i1 icmp eq (i32* getelementptr inbounds ([1 x i32], [1 x i32]* @gv2, i32 0, i32 0), i32* @gv1)
}
; PLAIN: gv_gep_vs_gv
; PLAIN: ret i1 false

define i1 @gv_gep_vs_gv_gep() {
  ret i1 icmp eq (i32* getelementptr inbounds ([1 x i32], [1 x i32]* @gv2, i32 0, i32 0), i32* getelementptr inbounds ([1 x i32], [1 x i32]* @gv3, i32 0, i32 0))
}
; PLAIN: gv_gep_vs_gv_gep
; PLAIN: ret i1 false

; CHECK: attributes #0 = { nounwind }
