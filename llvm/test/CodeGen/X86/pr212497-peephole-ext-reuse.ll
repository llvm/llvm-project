; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -O1 -verify-machineinstrs -o /dev/null

; Regression test for https://github.com/llvm/llvm-project/issues/212497
;
; PeepholeOpt's optimizeExtInstr must not reuse the result of a zero-extend for
; a same-block use that occurs *before* the extend. Doing so previously inserted
; `%x = COPY %ext.sub_8bit` ahead of `%ext = MOVZX ...`, which failed machine
; verification ("Virtual register defs don't dominate all uses").

@a = dso_local global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i8 0, align 1
@d = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4

declare void @f(i32 noundef)

define dso_local noundef i32 @main() local_unnamed_addr {
  %1 = load i32, ptr @b, align 4
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %19, label %3

3:
  %4 = load i32, ptr @d, align 4
  %5 = icmp eq i32 %4, 0
  %6 = zext i1 %5 to i32
  store i32 %6, ptr @e, align 4
  store i32 %6, ptr @b, align 4
  %7 = load i8, ptr @c, align 1
  %8 = zext i1 %5 to i8
  %9 = icmp eq i8 %7, %8
  %10 = select i1 %9, i32 2, i32 0
  br i1 %9, label %14, label %11

11:
  %12 = load volatile i32, ptr @a, align 4
  %13 = icmp ne i32 %12, 0
  br label %14

14:
  %15 = phi i1 [ true, %3 ], [ %13, %11 ]
  %16 = zext i1 %15 to i32
  store i32 %16, ptr @b, align 4
  tail call void @f(i32 noundef %10)
  %17 = load i32, ptr @b, align 4
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %3

19:
  ret i32 0
}
