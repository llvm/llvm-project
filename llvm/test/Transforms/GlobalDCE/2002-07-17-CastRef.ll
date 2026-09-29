; RUN: opt < %s -passes=globaldce
;
define internal void @func() {
        ret void
}

define void @main() {
        %X = addrspacecast ptr @func to ptr addrspace(1)             ; <i32*> [#uses=0]
        ret void
}

