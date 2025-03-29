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
  store i32 4, ptr %b.i, align 4
  %constexpr = ptrtoint ptr @_ZN1B1iEv to i32
  %constexpr1 = zext i32 %constexpr to i64
  %and1 = and i64 %constexpr1, 4294967296
  %cmp1 = icmp eq i64 %and1, 0
  br i1 %cmp1, label %phi.constexpr, label %cond_true.i

cond_true.i:                                      ; preds = %entry
  %ctg23.i = getelementptr i8, ptr %b.i, i32 0
  %tmp15.i = load ptr, ptr %ctg23.i, align 8
  %constexpr2 = ptrtoint ptr @_ZN1B1iEv to i32
  %ctg2.i = getelementptr i8, ptr %tmp15.i, i32 %constexpr2
  %tmp22.i = load ptr, ptr %ctg2.i, align 8
  br label %_Z3fooiM1BFvvE.exit

phi.constexpr:                                    ; preds = %entry
  %constexpr3 = ptrtoint ptr @_ZN1B1iEv to i32
  %constexpr4 = inttoptr i32 %constexpr3 to ptr
  br label %_Z3fooiM1BFvvE.exit

_Z3fooiM1BFvvE.exit:                              ; preds = %phi.constexpr, %cond_true.i
  %iftmp.2.0.i = phi ptr [ %tmp22.i, %cond_true.i ], [ %constexpr4, %phi.constexpr ]
  %ctg25.i = getelementptr i8, ptr %b.i, i32 0
  call void %iftmp.2.0.i(ptr %ctg25.i)
  store i32 6, ptr %b.i29, align 4
  %constexpr5 = ptrtoint ptr @_ZN1B1iEv to i32
  %constexpr6 = zext i32 %constexpr5 to i64
  %and2 = and i64 %constexpr6, 4294967296
  %cmp2 = icmp eq i64 %and2, 0
  br i1 %cmp2, label %phi.constexpr8, label %cond_true.i46

cond_true.i46:                                    ; preds = %_Z3fooiM1BFvvE.exit
  %ctg23.i36 = getelementptr i8, ptr %b.i29, i32 0
  %tmp15.i38 = load ptr, ptr %ctg23.i36, align 8
  %constexpr7 = ptrtoint ptr @_ZN1B1jEv to i32
  %ctg2.i42 = getelementptr i8, ptr %tmp15.i38, i32 %constexpr7
  %tmp22.i44 = load ptr, ptr %ctg2.i42, align 8
  br label %_Z3fooiM1BFvvE.exit56

phi.constexpr8:                                   ; preds = %_Z3fooiM1BFvvE.exit
  %constexpr9 = ptrtoint ptr @_ZN1B1jEv to i32
  %constexpr10 = inttoptr i32 %constexpr9 to ptr
  br label %_Z3fooiM1BFvvE.exit56

_Z3fooiM1BFvvE.exit56:                            ; preds = %phi.constexpr8, %cond_true.i46
  %iftmp.2.0.i49 = phi ptr [ %tmp22.i44, %cond_true.i46 ], [ %constexpr10, %phi.constexpr8 ]
  %ctg25.i54 = getelementptr i8, ptr %b.i29, i32 0
  call void %iftmp.2.0.i49(ptr %ctg25.i54)
  store i32 -1, ptr %b.i1, align 4
  %constexpr11 = ptrtoint ptr @_ZN1B1iEv to i32
  %constexpr12 = zext i32 %constexpr11 to i64
  %and3 = and i64 %constexpr12, 4294967296
  %cmp3 = icmp eq i64 %and3, 0
  br i1 %cmp3, label %phi.constexpr14, label %cond_true.i18

cond_true.i18:                                    ; preds = %_Z3fooiM1BFvvE.exit56
  %ctg23.i8 = getelementptr i8, ptr %b.i1, i32 0
  %tmp15.i10 = load ptr, ptr %ctg23.i8, align 8
  %constexpr13 = ptrtoint ptr @_ZN1B1iEv to i32
  %ctg2.i14 = getelementptr i8, ptr %tmp15.i10, i32 %constexpr13
  %tmp22.i16 = load ptr, ptr %ctg2.i14, align 8
  br label %_Z3fooiM1BFvvE.exit28

phi.constexpr14:                                  ; preds = %_Z3fooiM1BFvvE.exit56
  %constexpr15 = ptrtoint ptr @_ZN1B1iEv to i32
  %constexpr16 = inttoptr i32 %constexpr15 to ptr
  br label %_Z3fooiM1BFvvE.exit28

_Z3fooiM1BFvvE.exit28:                            ; preds = %phi.constexpr14, %cond_true.i18
  %iftmp.2.0.i21 = phi ptr [ %tmp22.i16, %cond_true.i18 ], [ %constexpr16, %phi.constexpr14 ]
  %ctg25.i26 = getelementptr i8, ptr %b.i1, i32 0
  call void %iftmp.2.0.i21(ptr %ctg25.i26)
  ret i32 0
}

; CHECK-NOT: LPC9

