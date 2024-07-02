; REQUIRES: x86-registered-target
; RUN: opt -passes=instcombine -S < %s | FileCheck %s

%class.Arr = type <{ [160 x %class.Derived], i32, [4 x i8] }>
%class.Derived = type { %class.Base, ptr }
%class.Base = type { ptr }

@array = hidden thread_local global %class.Arr zeroinitializer, align 32
; CHECK: @array{{.*}}align 32

@_ZTV7Derived = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr null, ptr null] }, align 8

define internal fastcc void @foo() unnamed_addr {
entry:
  store <8 x ptr> <ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV7Derived, i64 0, i32 0, i64 2), ptr null, ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV7Derived, i64 0, i32 0, i64 2), ptr null, ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV7Derived, i64 0, i32 0, i64 2), ptr null, ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV7Derived, i64 0, i32 0, i64 2), ptr null>, ptr @array, align 32
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"MaxTLSAlign", i32 256}
