; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis
; PR3304

       %struct.c37304a__vrec = type { i8, %struct.c37304a__vrec___disc___XVN }
        %struct.c37304a__vrec___disc___XVN = type {
%struct.c37304a__vrec___disc___XVN___O }
        %struct.c37304a__vrec___disc___XVN___O = type {  }

define void @_ada_c37304a() {
entry:
        %v = alloca %struct.c37304a__vrec               ;
        %0 = getelementptr %struct.c37304a__vrec* %v, i32 0, i32 0             
        store i8 8, i8* %0, align 1
        unreachable
}
