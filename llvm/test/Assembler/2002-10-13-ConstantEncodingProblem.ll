; RUN: llvm-as < %s | llvm-dis

%Domain = type { %Domain**, %Domain* }
@D = global %Domain zeroinitializer             ; <%Domain*> [#uses=0]

