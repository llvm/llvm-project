; "PLAIN" - No optimizations. This tests the default target layout
; constant folder.
; RUN: opt -S -o - < %s | FileCheck --check-prefix=PLAIN %s

; "OPT" - Optimizations but no targetdata. This tests default target layout
; folding in the optimizers.
; RUN: opt -S -o - -passes='function(instcombine),globalopt' < %s | FileCheck --check-prefix=OPT %s

; "TO" - Optimizations and targetdata. This tests target-dependent
; folding in the optimizers.
; RUN: opt -S -o - -passes='function(instcombine),globalopt' -data-layout="e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64" < %s | FileCheck --check-prefix=TO %s

; "SCEV" - ScalarEvolution with default target layout
; RUN: opt -passes='print<scalar-evolution>' < %s -disable-output 2>&1 | FileCheck --check-prefix=SCEV %s


; The automatic constant folder in opt does not have targetdata access, so
; it can't fold gep arithmetic, in general. However, the constant folder run
; from instcombine and global opt can use targetdata.

; PLAIN: @G8 = global ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -1)
; PLAIN: @G1 = global ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -1)
; PLAIN: @F8 = global ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -2)
; PLAIN: @F1 = global ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -2)
; PLAIN: @H8 = global ptr getelementptr (i8, ptr null, i32 -1)
; PLAIN: @H1 = global ptr getelementptr (i1, ptr null, i32 -1)
; OPT: @G8 = local_unnamed_addr global ptr null
; OPT: @G1 = local_unnamed_addr global ptr null
; OPT: @F8 = local_unnamed_addr global ptr inttoptr (i64 -1 to ptr)
; OPT: @F1 = local_unnamed_addr global ptr inttoptr (i64 -1 to ptr)
; OPT: @H8 = local_unnamed_addr global ptr inttoptr (i64 -1 to ptr)
; OPT: @H1 = local_unnamed_addr global ptr inttoptr (i64 -1 to ptr)
; TO: @G8 = local_unnamed_addr global ptr null
; TO: @G1 = local_unnamed_addr global ptr null
; TO: @F8 = local_unnamed_addr global ptr inttoptr (i64 -1 to ptr)
; TO: @F1 = local_unnamed_addr global ptr inttoptr (i64 -1 to ptr)
; TO: @H8 = local_unnamed_addr global ptr inttoptr (i64 -1 to ptr)
; TO: @H1 = local_unnamed_addr global ptr inttoptr (i64 -1 to ptr)

@G8 = global ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -1)
@G1 = global ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -1)
@F8 = global ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -2)
@F1 = global ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -2)
@H8 = global ptr getelementptr (i8, ptr inttoptr (i32 0 to ptr), i32 -1)
@H1 = global ptr getelementptr (i1, ptr inttoptr (i32 0 to ptr), i32 -1)

; The target-independent folder should be able to do some clever
; simplifications on sizeof, alignof, and offsetof expressions. The
; target-dependent folder should fold these down to constants.

; PLAIN: @a = constant i64 mul (i64 ptrtoint (ptr getelementptr ({ [7 x double], [7 x double] }, ptr null, i64 11) to i64), i64 15)
; PLAIN: @b = constant i64 ptrtoint (ptr getelementptr ({ i1, [13 x double] }, ptr null, i64 0, i32 1) to i64)
; PLAIN: @c = constant i64 ptrtoint (ptr getelementptr ({ double, double, double, double }, ptr null, i64 0, i32 2) to i64)
; PLAIN: @d = constant i64 ptrtoint (ptr getelementptr ([13 x double], ptr null, i64 0, i32 11) to i64)
; PLAIN: @e = constant i64 ptrtoint (ptr getelementptr ({ double, float, double, double }, ptr null, i64 0, i32 2) to i64)
; PLAIN: @f = constant i64 ptrtoint (ptr getelementptr ({ i1, <{ i16, i128 }> }, ptr null, i64 0, i32 1) to i64)
; PLAIN: @g = constant i64 ptrtoint (ptr getelementptr ({ i1, { double, double } }, ptr null, i64 0, i32 1) to i64)
; PLAIN: @h = constant i64 ptrtoint (ptr getelementptr (ptr, ptr null, i64 1) to i64)
; PLAIN: @i = constant i64 ptrtoint (ptr getelementptr ({ i1, ptr }, ptr null, i64 0, i32 1) to i64)
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

@a = constant i64 mul (i64 3, i64 mul (i64 ptrtoint (ptr getelementptr ({[7 x double], [7 x double]}, ptr null, i64 11) to i64), i64 5))
@b = constant i64 ptrtoint (ptr getelementptr ({i1, [13 x double]}, ptr null, i64 0, i32 1) to i64)
@c = constant i64 ptrtoint (ptr getelementptr ({double, double, double, double}, ptr null, i64 0, i32 2) to i64)
@d = constant i64 ptrtoint (ptr getelementptr ([13 x double], ptr null, i64 0, i32 11) to i64)
@e = constant i64 ptrtoint (ptr getelementptr ({double, float, double, double}, ptr null, i64 0, i32 2) to i64)
@f = constant i64 ptrtoint (ptr getelementptr ({i1, <{ i16, i128 }>}, ptr null, i64 0, i32 1) to i64)
@g = constant i64 ptrtoint (ptr getelementptr ({i1, {double, double}}, ptr null, i64 0, i32 1) to i64)
@h = constant i64 ptrtoint (ptr getelementptr (ptr, ptr null, i64 1) to i64)
@i = constant i64 ptrtoint (ptr getelementptr ({i1, ptr}, ptr null, i64 0, i32 1) to i64)

; The target-dependent folder should cast GEP indices to integer-sized pointers.

; PLAIN: @M = constant ptr getelementptr (i64, ptr null, i32 1)
; PLAIN: @N = constant ptr getelementptr ({ i64, i64 }, ptr null, i32 0, i32 1)
; PLAIN: @O = constant ptr getelementptr ([2 x i64], ptr null, i32 0, i32 1)
; OPT: @M = local_unnamed_addr constant ptr inttoptr (i64 8 to ptr)
; OPT: @N = local_unnamed_addr constant ptr inttoptr (i64 8 to ptr)
; OPT: @O = local_unnamed_addr constant ptr inttoptr (i64 8 to ptr)
; TO: @M = local_unnamed_addr constant ptr inttoptr (i64 8 to ptr)
; TO: @N = local_unnamed_addr constant ptr inttoptr (i64 8 to ptr)
; TO: @O = local_unnamed_addr constant ptr inttoptr (i64 8 to ptr)

@M = constant ptr getelementptr (i64, ptr null, i32 1)
@N = constant ptr getelementptr ({ i64, i64 }, ptr null, i32 0, i32 1)
@O = constant ptr getelementptr ([2 x i64], ptr null, i32 0, i32 1)

; Fold GEP of a GEP. Very simple cases are folded without targetdata.

; PLAIN: @Y = global ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 2)
; PLAIN: @Z = global ptr getelementptr inbounds (i32, ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 0), i64 1)
; OPT: @Y = local_unnamed_addr global ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 2)
; OPT: @Z = local_unnamed_addr global ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 1)
; TO: @Y = local_unnamed_addr global ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 2)
; TO: @Z = local_unnamed_addr global ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 1)

@ext = external global [3 x { i32, i32 }]
@Y = global ptr getelementptr inbounds ([3 x { i32, i32 }], ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 1), i64 1)
@Z = global ptr getelementptr inbounds (i32, ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 0), i64 1)

; Duplicate all of the above as function return values rather than
; global initializers.

; PLAIN: define ptr @goo8() #0 {
; PLAIN:   %t = bitcast ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -1) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; PLAIN: define ptr @goo1() #0 {
; PLAIN:   %t = bitcast ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -1) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; PLAIN: define ptr @foo8() #0 {
; PLAIN:   %t = bitcast ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -2) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; PLAIN: define ptr @foo1() #0 {
; PLAIN:   %t = bitcast ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -2) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; PLAIN: define ptr @hoo8() #0 {
; PLAIN:   %t = bitcast ptr getelementptr (i8, ptr null, i32 -1) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; PLAIN: define ptr @hoo1() #0 {
; PLAIN:   %t = bitcast ptr getelementptr (i1, ptr null, i32 -1) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; OPT: define ptr @goo8() local_unnamed_addr #0 {
; OPT:   ret ptr null
; OPT: }
; OPT: define ptr @goo1() local_unnamed_addr #0 {
; OPT:   ret ptr null
; OPT: }
; OPT: define ptr @foo8() local_unnamed_addr #0 {
; OPT:   ret ptr inttoptr (i64 -1 to ptr)
; OPT: }
; OPT: define ptr @foo1() local_unnamed_addr #0 {
; OPT:   ret ptr inttoptr (i64 -1 to ptr)
; OPT: }
; OPT: define ptr @hoo8() local_unnamed_addr #0 {
; OPT:   ret ptr inttoptr (i64 -1 to ptr)
; OPT: }
; OPT: define ptr @hoo1() local_unnamed_addr #0 {
; OPT:   ret ptr inttoptr (i64 -1 to ptr)
; OPT: }
; TO: define ptr @goo8() local_unnamed_addr #0 {
; TO:   ret ptr null
; TO: }
; TO: define ptr @goo1() local_unnamed_addr #0 {
; TO:   ret ptr null
; TO: }
; TO: define ptr @foo8() local_unnamed_addr #0 {
; TO:   ret ptr inttoptr (i64 -1 to ptr)
; TO: }
; TO: define ptr @foo1() local_unnamed_addr #0 {
; TO:   ret ptr inttoptr (i64 -1 to ptr)
; TO: }
; TO: define ptr @hoo8() local_unnamed_addr #0 {
; TO:   ret ptr inttoptr (i64 -1 to ptr)
; TO: }
; TO: define ptr @hoo1() local_unnamed_addr #0 {
; TO:   ret ptr inttoptr (i64 -1 to ptr)
; TO: }
; SCEV: Classifying expressions for: @goo8
; SCEV:   %t = bitcast ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -1) to ptr
; SCEV:   -->  (-1 + inttoptr (i32 1 to ptr))
; SCEV: Classifying expressions for: @goo1
; SCEV:   %t = bitcast ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -1) to ptr
; SCEV:   -->  (-1 + inttoptr (i32 1 to ptr))
; SCEV: Classifying expressions for: @foo8
; SCEV:   %t = bitcast ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -2) to ptr
; SCEV:   -->  (-2 + inttoptr (i32 1 to ptr))
; SCEV: Classifying expressions for: @foo1
; SCEV:   %t = bitcast ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -2) to ptr
; SCEV:   -->  (-2 + inttoptr (i32 1 to ptr))
; SCEV: Classifying expressions for: @hoo8
; SCEV:   -->  (-1 + null)<nuw><nsw> U: [-1,0) S: [-1,0)
; SCEV: Classifying expressions for: @hoo1
; SCEV:   -->  (-1 + null)<nuw><nsw> U: [-1,0) S: [-1,0)

define ptr @goo8() nounwind {
  %t = bitcast ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -1) to ptr
  ret ptr %t
}
define ptr @goo1() nounwind {
  %t = bitcast ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -1) to ptr
  ret ptr %t
}
define ptr @foo8() nounwind {
  %t = bitcast ptr getelementptr (i8, ptr inttoptr (i32 1 to ptr), i32 -2) to ptr
  ret ptr %t
}
define ptr @foo1() nounwind {
  %t = bitcast ptr getelementptr (i1, ptr inttoptr (i32 1 to ptr), i32 -2) to ptr
  ret ptr %t
}
define ptr @hoo8() nounwind {
  %t = bitcast ptr getelementptr (i8, ptr inttoptr (i32 0 to ptr), i32 -1) to ptr
  ret ptr %t
}
define ptr @hoo1() nounwind {
  %t = bitcast ptr getelementptr (i1, ptr inttoptr (i32 0 to ptr), i32 -1) to ptr
  ret ptr %t
}

; PLAIN: define i64 @fa() #0 {
; PLAIN:   %t = bitcast i64 mul (i64 ptrtoint (ptr getelementptr ({ [7 x double], [7 x double] }, ptr null, i64 11) to i64), i64 15) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fb() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ i1, [13 x double] }, ptr null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fc() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ double, double, double, double }, ptr null, i64 0, i32 2) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fd() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (ptr getelementptr ([13 x double], ptr null, i64 0, i32 11) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fe() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ double, float, double, double }, ptr null, i64 0, i32 2) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @ff() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ i1, <{ i16, i128 }> }, ptr null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fg() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ i1, { double, double } }, ptr null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fh() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fi() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ i1, ptr }, ptr null, i64 0, i32 1) to i64) to i64
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
; SCEV-LABEL: Classifying expressions for: @fa
; SCEV:   %t = bitcast i64 mul (i64 ptrtoint (ptr getelementptr ({ [7 x double], [7 x double] }, ptr null, i64 11) to i64), i64 15) to i64
; SCEV:   -->  18480
; SCEV-LABEL: Classifying expressions for: @fb
; SCEV:  %t = bitcast i64 ptrtoint (ptr getelementptr ({ i1, [13 x double] }, ptr null, i64 0, i32 1) to i64) to i64
; SCEV:   -->  8
; SCEV-LABEL: Classifying expressions for: @fc
; SCEV:  %t = bitcast i64 ptrtoint (ptr getelementptr ({ double, double, double, double }, ptr null, i64 0, i32 2) to i64) to i64
; SCEV:   -->  16
; SCEV-LABEL: Classifying expressions for: @fd
; SCEV:   %t = bitcast i64 ptrtoint (ptr getelementptr ([13 x double], ptr null, i64 0, i32 11) to i64) to i64
; SCEV:   -->  88
; SCEV-LABEL: Classifying expressions for: @fe
; SCEV:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ double, float, double, double }, ptr null, i64 0, i32 2) to i64) to i64
; SCEV:   -->  16
; SCEV-LABEL: Classifying expressions for: @ff
; SCEV:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ i1, <{ i16, i128 }> }, ptr null, i64 0, i32 1) to i64) to i64
; SCEV:   -->  1
; SCEV-LABEL: Classifying expressions for: @fg
; SCEV:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ i1, { double, double } }, ptr null, i64 0, i32 1) to i64) to i64
; SCEV:   -->  8
; SCEV-LABEL: Classifying expressions for: @fh
; SCEV:   %t = bitcast i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64) to i64
; SCEV:   --> 8
; SCEV-LABEL: Classifying expressions for: @fi
; SCEV:   %t = bitcast i64 ptrtoint (ptr getelementptr ({ i1, ptr }, ptr null, i64 0, i32 1) to i64) to i64
; SCEV:   --> 8

define i64 @fa() nounwind {
  %t = bitcast i64 mul (i64 3, i64 mul (i64 ptrtoint (ptr getelementptr ({[7 x double], [7 x double]}, ptr null, i64 11) to i64), i64 5)) to i64
  ret i64 %t
}
define i64 @fb() nounwind {
  %t = bitcast i64 ptrtoint (ptr getelementptr ({i1, [13 x double]}, ptr null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fc() nounwind {
  %t = bitcast i64 ptrtoint (ptr getelementptr ({double, double, double, double}, ptr null, i64 0, i32 2) to i64) to i64
  ret i64 %t
}
define i64 @fd() nounwind {
  %t = bitcast i64 ptrtoint (ptr getelementptr ([13 x double], ptr null, i64 0, i32 11) to i64) to i64
  ret i64 %t
}
define i64 @fe() nounwind {
  %t = bitcast i64 ptrtoint (ptr getelementptr ({double, float, double, double}, ptr null, i64 0, i32 2) to i64) to i64
  ret i64 %t
}
define i64 @ff() nounwind {
  %t = bitcast i64 ptrtoint (ptr getelementptr ({i1, <{ i16, i128 }>}, ptr null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fg() nounwind {
  %t = bitcast i64 ptrtoint (ptr getelementptr ({i1, {double, double}}, ptr null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fh() nounwind {
  %t = bitcast i64 ptrtoint (ptr getelementptr (ptr, ptr null, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fi() nounwind {
  %t = bitcast i64 ptrtoint (ptr getelementptr ({i1, ptr}, ptr null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}

; PLAIN: define ptr @fM() #0 {
; PLAIN:   %t = bitcast ptr getelementptr (i64, ptr null, i32 1) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; PLAIN: define ptr @fN() #0 {
; PLAIN:   %t = bitcast ptr getelementptr ({ i64, i64 }, ptr null, i32 0, i32 1) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; PLAIN: define ptr @fO() #0 {
; PLAIN:   %t = bitcast ptr getelementptr ([2 x i64], ptr null, i32 0, i32 1) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; OPT: define ptr @fM() local_unnamed_addr #0 {
; OPT:   ret ptr inttoptr (i64 8 to ptr)
; OPT: }
; OPT: define ptr @fN() local_unnamed_addr #0 {
; OPT:   ret ptr inttoptr (i64 8 to ptr)
; OPT: }
; OPT: define ptr @fO() local_unnamed_addr #0 {
; OPT:   ret ptr inttoptr (i64 8 to ptr)
; OPT: }
; TO: define ptr @fM() local_unnamed_addr #0 {
; TO:   ret ptr inttoptr (i64 8 to ptr)
; TO: }
; TO: define ptr @fN() local_unnamed_addr #0 {
; TO:   ret ptr inttoptr (i64 8 to ptr)
; TO: }
; TO: define ptr @fO() local_unnamed_addr #0 {
; TO:   ret ptr inttoptr (i64 8 to ptr)
; TO: }
; SCEV: Classifying expressions for: @fM
; SCEV:   %t = bitcast ptr getelementptr (i64, ptr null, i32 1) to ptr
; SCEV:    --> (8 + null)<nuw><nsw> U: [8,9) S: [8,9)
; SCEV: Classifying expressions for: @fN
; SCEV:   %t = bitcast ptr getelementptr ({ i64, i64 }, ptr null, i32 0, i32 1) to ptr
; SCEV:   --> (8 + null)<nuw><nsw> U: [8,9) S: [8,9)
; SCEV: Classifying expressions for: @fO
; SCEV:   %t = bitcast ptr getelementptr ([2 x i64], ptr null, i32 0, i32 1) to ptr
; SCEV:   --> (8 + null)<nuw><nsw> U: [8,9) S: [8,9)

define ptr @fM() nounwind {
  %t = bitcast ptr getelementptr (i64, ptr null, i32 1) to ptr
  ret ptr %t
}
define ptr @fN() nounwind {
  %t = bitcast ptr getelementptr ({ i64, i64 }, ptr null, i32 0, i32 1) to ptr
  ret ptr %t
}
define ptr @fO() nounwind {
  %t = bitcast ptr getelementptr ([2 x i64], ptr null, i32 0, i32 1) to ptr
  ret ptr %t
}

; PLAIN: define ptr @fZ() #0 {
; PLAIN:   %t = bitcast ptr getelementptr inbounds (i32, ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 0), i64 1) to ptr
; PLAIN:   ret ptr %t
; PLAIN: }
; OPT: define ptr @fZ() local_unnamed_addr #0 {
; OPT:   ret ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 1)
; OPT: }
; TO: define ptr @fZ() local_unnamed_addr #0 {
; TO:   ret ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 1)
; TO: }
; SCEV: Classifying expressions for: @fZ
; SCEV:   %t = bitcast ptr getelementptr inbounds (i32, ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 0), i64 1) to ptr
; SCEV:   -->  (12 + @ext)

define ptr @fZ() nounwind {
  %t = bitcast ptr getelementptr inbounds (i32, ptr getelementptr inbounds ([3 x { i32, i32 }], ptr @ext, i64 0, i64 1, i32 0), i64 1) to ptr
  ret ptr %t
}

; PR15262 - Check GEP folding with casts between address spaces.

@p0 = global [4 x i8] zeroinitializer, align 1
@p12 = addrspace(12) global [4 x i8] zeroinitializer, align 1

define ptr @different_addrspace() nounwind noinline {
; OPT: different_addrspace
  %p = getelementptr inbounds i8, ptr addrspacecast (ptr addrspace(12) @p12 to ptr),
                                  i32 2
  ret ptr %p
; OPT: ret ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(12) @p12 to ptr), i64 2)
}

define ptr @same_addrspace() nounwind noinline {
; OPT: same_addrspace
  %p = getelementptr inbounds i8, ptr @p0, i32 2
  ret ptr %p
; OPT: ret ptr getelementptr inbounds ([4 x i8], ptr @p0, i64 0, i64 2)
}

@gv1 = internal global i32 1
@gv2 = internal global [1 x i32] [ i32 2 ]
@gv3 = internal global [1 x i32] [ i32 2 ]

; Handled by TI-independent constant folder
define i1 @gv_gep_vs_gv() {
  ret i1 icmp eq (ptr @gv2, ptr @gv1)
}
; PLAIN: gv_gep_vs_gv
; PLAIN: ret i1 false

define i1 @gv_gep_vs_gv_gep() {
  ret i1 icmp eq (ptr @gv2, ptr @gv3)
}
; PLAIN: gv_gep_vs_gv_gep
; PLAIN: ret i1 false

; CHECK: attributes #0 = { nounwind }
