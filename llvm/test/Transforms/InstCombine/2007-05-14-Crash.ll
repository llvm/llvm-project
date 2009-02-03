; RUN: llvm-as < %s | opt -instcombine -disable-output

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "powerpc-apple-darwin8.8.0"        

%struct.abc = type { i32, [32 x i8] }        
%struct.def = type { i8**, %struct.abc }        
        %struct.anon = type <{  }>

define i8* @foo(%struct.anon* %deviceRef, %struct.abc* %pCap) {
entry:
        %tmp1 = bitcast %struct.anon* %deviceRef to %struct.def*            
        %tmp3 = getelementptr %struct.def* %tmp1, i32 0, i32 1               
        %tmp35 = bitcast %struct.abc* %tmp3 to i8*           
        ret i8* %tmp35
}


