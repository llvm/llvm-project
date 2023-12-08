; RUN: opt -passes=always-inline -S %s | FileCheck %s
%0 = type { i64, i64, ptr addrspace(1), ptr addrspace(1) }
%__aaa_struct = type { { ptr, i32, i32, ptr, ptr addrspace(1) }, %0, [17 x i8], { ptr, i32, i32, ptr, ptr addrspace(1) }, %0, [18 x i8] }
%struct.__block_descriptor = type { i64, i64 }
%struct.__block_literal_generic = type { ptr, i32, i32, ptr, ptr addrspace(1) }

@__aaa_struct_ptr = external addrspace(1) global %__aaa_struct
@__aaa_const_init = constant %__aaa_struct { { ptr, i32, i32, ptr, ptr addrspace(1) } { ptr null, i32 1342177280, i32 0, ptr @bl0_block_invoke, ptr addrspace(1) getelementptr inbounds (%__aaa_struct, ptr addrspace(1) @__aaa_struct_ptr, i32 0, i32 1) }, %0 { i64 0, i64 32, ptr addrspace(1) getelementptr inbounds (%__aaa_struct, ptr addrspace(1) @__aaa_struct_ptr, i32 0, i32 2, i32 0), ptr addrspace(1) null }, [17 x i8] c"bl0_block_invoke\00", { ptr, i32, i32, ptr, ptr addrspace(1) } { ptr null, i32 1342177280, i32 0, ptr @__f1_block_invoke, ptr addrspace(1) getelementptr inbounds (%__aaa_struct, ptr addrspace(1) @__aaa_struct_ptr, i32 0, i32 4) }, %0 { i64 0, i64 32, ptr addrspace(1) getelementptr inbounds (%__aaa_struct, ptr addrspace(1) @__aaa_struct_ptr, i32 0, i32 5, i32 0), ptr addrspace(1) null }, [18 x i8] c"__f1_block_invoke\00" }

; Function Attrs: alwaysinline norecurse nounwind readonly
define i32 @bl0_block_invoke(ptr addrspace(4) nocapture readnone, ptr addrspace(1) nocapture readonly) #0 {
entry:
  %2 = load i32, ptr addrspace(1) %1, align 4
  %mul = shl nsw i32 %2, 1
  ret i32 %mul
}

; Function Attrs: alwaysinline nounwind
define i32 @f0(ptr addrspace(1), ptr addrspace(4)) #1 {
entry:
  %2 = getelementptr inbounds %struct.__block_literal_generic, ptr addrspace(4) %1, i64 0, i32 3
  %3 = load ptr, ptr addrspace(4) %2, align 8
  %call = tail call i32 %3(ptr addrspace(4) %1, ptr addrspace(1) %0) #2
  ret i32 %call
}

; CHECK-LABEL: define void @f1
; CHECK: %1 = load ptr, ptr addrspace(4) getelementptr inbounds (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) @__aaa_struct_ptr to ptr addrspace(4)), i64 16), align 8

; Function Attrs: alwaysinline nounwind
define void @f1(ptr addrspace(1)) #1 {
entry:
  %call = tail call i32 @f0(ptr addrspace(1) %0, ptr addrspace(4) addrspacecast (ptr addrspace(1) @__aaa_struct_ptr to ptr addrspace(4))) #3
  store i32 %call, ptr addrspace(1) %0, align 4
  %call1 = tail call i32 @f0(ptr addrspace(1) %0, ptr addrspace(4) addrspacecast (ptr addrspace(1) getelementptr inbounds (%__aaa_struct, ptr addrspace(1) @__aaa_struct_ptr, i32 0, i32 3) to ptr addrspace(4))) #3
  store i32 %call1, ptr addrspace(1) %0, align 4
  ret void
}

; Function Attrs: alwaysinline norecurse nounwind readonly
define i32 @__f1_block_invoke(ptr addrspace(4) nocapture readnone, ptr addrspace(1) nocapture readonly) #0 {
entry:
  %2 = load i32, ptr addrspace(1) %1, align 4
  %add = add nsw i32 %2, 1
  ret i32 %add
}

attributes #0 = { alwaysinline norecurse nounwind readonly }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { nobuiltin nounwind }
attributes #3 = { nobuiltin }
