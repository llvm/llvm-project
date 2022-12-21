; RUN: llc -mtriple arm-apple-darwin -relocation-model pic -filetype asm -o - %s | FileCheck %s

%struct.B = type { i32 }
%struct.anon = type { ptr, i32 }
@str = internal constant [7 x i8] c"i, %d\0A\00"
@str1 = internal constant [7 x i8] c"j, %d\0A\00"

define internal void @_ZN1B1iEv(ptr %this) {
entry:
  %tmp2 = load i32, ptr %this
  %tmp4 = tail call i32 (ptr, ...) @printf(ptr @str, i32 %tmp2)
  ret void
}

declare i32 @printf(ptr, ...)

define internal void @_ZN1B1jEv(ptr %this) {
entry:
  %tmp2 = load i32, ptr %this
  %tmp4 = tail call i32 (ptr, ...) @printf(ptr @str1, i32 %tmp2)
  ret void
}

define i32 @main() {
entry:
  %b.i29 = alloca %struct.B, align 4
  %b.i1 = alloca %struct.B, align 4
  %b.i = alloca %struct.B, align 4
  store i32 4, ptr %b.i
  br i1 icmp eq (i64 and (i64 zext (i32 ptrtoint (ptr @_ZN1B1iEv to i32) to i64), i64 4294967296), i64 0), label %_Z3fooiM1BFvvE.exit, label %cond_true.i

cond_true.i:
  %ctg23.i = getelementptr i8, ptr %b.i, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (ptr @_ZN1B1iEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp15.i = load ptr, ptr %ctg23.i
  %ctg2.i = getelementptr i8, ptr %tmp15.i, i32 ptrtoint (ptr @_ZN1B1iEv to i32)
  %tmp22.i = load ptr, ptr %ctg2.i
  br label %_Z3fooiM1BFvvE.exit

_Z3fooiM1BFvvE.exit:
  %iftmp.2.0.i = phi ptr [ %tmp22.i, %cond_true.i ], [ inttoptr (i32 ptrtoint (ptr @_ZN1B1iEv to i32) to ptr), %entry ]
  %ctg25.i = getelementptr i8, ptr %b.i, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (ptr @_ZN1B1iEv to i32) to i64), i64 32) to i32), i32 1)
  call void %iftmp.2.0.i(ptr %ctg25.i)
  store i32 6, ptr %b.i29
  br i1 icmp eq (i64 and (i64 zext (i32 ptrtoint (ptr @_ZN1B1jEv to i32) to i64), i64 4294967296), i64 0), label %_Z3fooiM1BFvvE.exit56, label %cond_true.i46

cond_true.i46:
  %ctg23.i36 = getelementptr i8, ptr %b.i29, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (ptr @_ZN1B1jEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp15.i38 = load ptr, ptr %ctg23.i36
  %ctg2.i42 = getelementptr i8, ptr %tmp15.i38, i32 ptrtoint (ptr @_ZN1B1jEv to i32)
  %tmp22.i44 = load ptr, ptr %ctg2.i42
  br label %_Z3fooiM1BFvvE.exit56

_Z3fooiM1BFvvE.exit56:
  %iftmp.2.0.i49 = phi ptr [ %tmp22.i44, %cond_true.i46 ], [ inttoptr (i32 ptrtoint (ptr @_ZN1B1jEv to i32) to ptr), %_Z3fooiM1BFvvE.exit ]
  %ctg25.i54 = getelementptr i8, ptr %b.i29, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (ptr @_ZN1B1jEv to i32) to i64), i64 32) to i32), i32 1)
  call void %iftmp.2.0.i49(ptr %ctg25.i54)
  store i32 -1, ptr %b.i1
  br i1 icmp eq (i64 and (i64 zext (i32 ptrtoint (ptr @_ZN1B1iEv to i32) to i64), i64 4294967296), i64 0), label %_Z3fooiM1BFvvE.exit28, label %cond_true.i18

cond_true.i18:
  %ctg23.i8 = getelementptr i8, ptr %b.i1, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (ptr @_ZN1B1iEv to i32) to i64), i64 32) to i32), i32 1)
  %tmp15.i10 = load ptr, ptr %ctg23.i8
  %ctg2.i14 = getelementptr i8, ptr %tmp15.i10, i32 ptrtoint (ptr @_ZN1B1iEv to i32)
  %tmp22.i16 = load ptr, ptr %ctg2.i14
  br label %_Z3fooiM1BFvvE.exit28

_Z3fooiM1BFvvE.exit28:
  %iftmp.2.0.i21 = phi ptr [ %tmp22.i16, %cond_true.i18 ], [ inttoptr (i32 ptrtoint (ptr @_ZN1B1iEv to i32) to ptr), %_Z3fooiM1BFvvE.exit56 ]
  %ctg25.i26 = getelementptr i8, ptr %b.i1, i32 ashr (i32 trunc (i64 lshr (i64 zext (i32 ptrtoint (ptr @_ZN1B1iEv to i32) to i64), i64 32) to i32), i32 1)
  call void %iftmp.2.0.i21(ptr %ctg25.i26)
  ret i32 0
}

; CHECK-NOT: LPC9

