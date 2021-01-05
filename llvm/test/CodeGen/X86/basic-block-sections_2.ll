; Check that the basic block sections suffix naming does not conflict with __cxx_global_var_init.N naming.
; How to  generate this file:
;; class A {
;;  public:
;;    A(bool a) { }
;; };
;;
;; extern bool bar(int);
;; A g_a(bar(5) ? bar(3) : false), g_b(true);
;;
;; $ clang -O1 -emit-llvm -S
;;
; __cxx_global_var_init has multiple basic blocks which will produce many sections.
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=all -unique-basic-block-section-names | FileCheck %s

; CHECK-LABEL: __cxx_global_var_init:
; CHECK-LABEL: __cxx_global_var_init.__part.1:
; CHECK-LABEL: __cxx_global_var_init.1:

%class.A = type { i8 }

$_ZN1AC2Eb = comdat any

@g_a = dso_local global %class.A zeroinitializer, align 1
@g_b = dso_local global %class.A zeroinitializer, align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_symcollision.cc, i8* null }]

define internal fastcc void @__cxx_global_var_init() unnamed_addr section ".text.startup" {
entry:
  %call = call zeroext i1 @_Z3bari(i32 5)
  br i1 %call, label %cond.true, label %cond.end

cond.true:                                        ; preds = %entry
  %call1 = call zeroext i1 @_Z3bari(i32 3)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i1 [ %call1, %cond.true ], [ false, %entry ]
  call void @_ZN1AC2Eb(%class.A* nonnull @g_a, i1 zeroext %cond)
  ret void
}

declare dso_local zeroext i1 @_Z3bari(i32) local_unnamed_addr

define linkonce_odr dso_local void @_ZN1AC2Eb(%class.A* %this, i1 zeroext %a) unnamed_addr comdat align 2 {
entry:
  ret void
}

define internal fastcc void @__cxx_global_var_init.1() unnamed_addr  section ".text.startup" {
entry:
  call void @_ZN1AC2Eb(%class.A* nonnull @g_b, i1 zeroext true)
  ret void
}

define internal void @_GLOBAL__sub_I_symcollision.cc()  section ".text.startup" {
entry:
  call fastcc void @__cxx_global_var_init()
  call fastcc void @__cxx_global_var_init.1()
  ret void
}
