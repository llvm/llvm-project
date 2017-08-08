; RUN: opt -always-inline -S %s | FileCheck %s
%0 = type { i64, i64, i8 addrspace(1)*, i8 addrspace(1)* }
%__aaa_struct = type { { i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(1)* }, %0, [17 x i8], { i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(1)* }, %0, [18 x i8] }
%struct.__block_descriptor = type { i64, i64 }
%struct.__block_literal_generic = type { i8*, i32, i32, i8*, %struct.__block_descriptor addrspace(1)* }

@__aaa_struct_ptr = external addrspace(1) global %__aaa_struct
@__aaa_const_init = constant %__aaa_struct { { i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(1)* } { i8** null, i32 1342177280, i32 0, i8* bitcast (i32 (i8 addrspace(4)*, i32 addrspace(1)*)* @bl0_block_invoke to i8*), %struct.__block_descriptor addrspace(1)* bitcast (%0 addrspace(1)* getelementptr inbounds (%__aaa_struct, %__aaa_struct addrspace(1)* @__aaa_struct_ptr, i32 0, i32 1) to %struct.__block_descriptor addrspace(1)*) }, %0 { i64 0, i64 32, i8 addrspace(1)* getelementptr inbounds (%__aaa_struct, %__aaa_struct addrspace(1)* @__aaa_struct_ptr, i32 0, i32 2, i32 0), i8 addrspace(1)* null }, [17 x i8] c"bl0_block_invoke\00", { i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(1)* } { i8** null, i32 1342177280, i32 0, i8* bitcast (i32 (i8 addrspace(4)*, i32 addrspace(1)*)* @__f1_block_invoke to i8*), %struct.__block_descriptor addrspace(1)* bitcast (%0 addrspace(1)* getelementptr inbounds (%__aaa_struct, %__aaa_struct addrspace(1)* @__aaa_struct_ptr, i32 0, i32 4) to %struct.__block_descriptor addrspace(1)*) }, %0 { i64 0, i64 32, i8 addrspace(1)* getelementptr inbounds (%__aaa_struct, %__aaa_struct addrspace(1)* @__aaa_struct_ptr, i32 0, i32 5, i32 0), i8 addrspace(1)* null }, [18 x i8] c"__f1_block_invoke\00" }

; Function Attrs: alwaysinline norecurse nounwind readonly
define i32 @bl0_block_invoke(i8 addrspace(4)* nocapture readnone, i32 addrspace(1)* nocapture readonly) #0 {
entry:
  %2 = load i32, i32 addrspace(1)* %1, align 4
  %mul = shl nsw i32 %2, 1
  ret i32 %mul
}

; Function Attrs: alwaysinline nounwind
define i32 @f0(i32 addrspace(1)*, i32 (i32 addrspace(1)*) addrspace(4)*) #1 {
entry:
  %block.literal = bitcast i32 (i32 addrspace(1)*) addrspace(4)* %1 to %struct.__block_literal_generic addrspace(4)*
  %2 = getelementptr inbounds %struct.__block_literal_generic, %struct.__block_literal_generic addrspace(4)* %block.literal, i64 0, i32 3
  %3 = bitcast i32 (i32 addrspace(1)*) addrspace(4)* %1 to i8 addrspace(4)*
  %4 = bitcast i8* addrspace(4)* %2 to i32 (i8 addrspace(4)*, i32 addrspace(1)*)* addrspace(4)*
  %5 = load i32 (i8 addrspace(4)*, i32 addrspace(1)*)*, i32 (i8 addrspace(4)*, i32 addrspace(1)*)* addrspace(4)* %4, align 8
  %call = tail call i32 %5(i8 addrspace(4)* %3, i32 addrspace(1)* %0) #2
  ret i32 %call
}

; CHECK-LABEL: define void @f1
; CHECK: %1 = load i32 (i8 addrspace(4)*, i32 addrspace(1)*)*, i32 (i8 addrspace(4)*, i32 addrspace(1)*)* addrspace(4)* bitcast (i8* addrspace(4)* getelementptr (%__aaa_struct, %__aaa_struct addrspace(4)* addrspacecast (%__aaa_struct addrspace(1)* @__aaa_struct_ptr to %__aaa_struct addrspace(4)*), i64 0, i32 0, i32 3) to i32 (i8 addrspace(4)*, i32 addrspace(1)*)* addrspace(4)*), align 8

; Function Attrs: alwaysinline nounwind
define void @f1(i32 addrspace(1)*) #1 {
entry:
  %call = tail call i32 @f0(i32 addrspace(1)* %0, i32 (i32 addrspace(1)*) addrspace(4)* addrspacecast (i32 (i32 addrspace(1)*) addrspace(1)* bitcast (%__aaa_struct addrspace(1)* @__aaa_struct_ptr to i32 (i32 addrspace(1)*) addrspace(1)*) to i32 (i32 addrspace(1)*) addrspace(4)*)) #3
  store i32 %call, i32 addrspace(1)* %0, align 4
  %call1 = tail call i32 @f0(i32 addrspace(1)* %0, i32 (i32 addrspace(1)*) addrspace(4)* addrspacecast (i32 (i32 addrspace(1)*) addrspace(1)* bitcast ({ i8**, i32, i32, i8*, %struct.__block_descriptor addrspace(1)* } addrspace(1)* getelementptr inbounds (%__aaa_struct, %__aaa_struct addrspace(1)* @__aaa_struct_ptr, i32 0, i32 3) to i32 (i32 addrspace(1)*) addrspace(1)*) to i32 (i32 addrspace(1)*) addrspace(4)*)) #3
  store i32 %call1, i32 addrspace(1)* %0, align 4
  ret void
}

; Function Attrs: alwaysinline norecurse nounwind readonly
define i32 @__f1_block_invoke(i8 addrspace(4)* nocapture readnone, i32 addrspace(1)* nocapture readonly) #0 {
entry:
  %2 = load i32, i32 addrspace(1)* %1, align 4
  %add = add nsw i32 %2, 1
  ret i32 %add
}

attributes #0 = { alwaysinline norecurse nounwind readonly }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { nobuiltin nounwind }
attributes #3 = { nobuiltin }
