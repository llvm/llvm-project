; RUN: opt --mtriple x86_64-unknown-linux-gnu < %s -passes="embed-bitcode<thinlto>" -S | FileCheck %s

; CHECK-NOT: $_ZTV3Foo = comdat any
$_ZTV3Foo = comdat any

$_ZTI3Foo = comdat any

; CHECK: @_ZTV3Foo = external hidden unnamed_addr constant { [5 x ptr] }, align 8
; CHECK: @_ZTI3Foo = linkonce_odr hidden constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS3Foo, ptr @_ZTISt13runtime_error }, comdat, align 8
; CHECK: @llvm.embedded.object = private constant {{.*}}, section ".llvm.lto", align 1
; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @llvm.embedded.object], section "llvm.metadata"
@_ZTV3Foo = linkonce_odr hidden unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI3Foo, ptr @_ZN3FooD2Ev, ptr @_ZN3FooD0Ev, ptr @_ZNKSt13runtime_error4whatEv] }, comdat, align 8, !type !0, !type !1, !type !2, !type !3, !type !4, !type !5
@_ZTI3Foo = linkonce_odr hidden constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS3Foo, ptr @_ZTISt13runtime_error }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTS3Foo = constant [5 x i8] c"3Foo\00"
@_ZTISt13runtime_error = external constant ptr

declare void @_ZN3FooD2Ev()

declare void @_ZN3FooD0Ev()

declare ptr @_ZNKSt13runtime_error4whatEv()

!llvm.module.flags = !{!6}

!0 = !{i64 16, !"_ZTS3Foo"}
!1 = !{i64 32, !"_ZTSM3FooKDoFPKcvE.virtual"}
!2 = !{i64 16, !"_ZTSSt13runtime_error"}
!3 = !{i64 32, !"_ZTSMSt13runtime_errorKDoFPKcvE.virtual"}
!4 = !{i64 16, !"_ZTSSt9exception"}
!5 = !{i64 32, !"_ZTSMSt9exceptionKDoFPKcvE.virtual"}
!6 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
