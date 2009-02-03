; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

declare void @foo1()

declare void @foo2()

define void @test1(i32 %V) {
        %C1 = icmp eq i32 %V, 4         ; <i1> [#uses=1]
        %C2 = icmp eq i32 %V, 17                ; <i1> [#uses=1]
        %CN = or i1 %C1, %C2            ; <i1> [#uses=1]
        br i1 %CN, label %T, label %F
T:              ; preds = %0
        call void @foo1( )
        ret void
F:              ; preds = %0
        call void @foo2( )
        ret void
}

define void @test2(i32 %V) {
        %C1 = icmp ne i32 %V, 4         ; <i1> [#uses=1]
        %C2 = icmp ne i32 %V, 17                ; <i1> [#uses=1]
        %CN = and i1 %C1, %C2           ; <i1> [#uses=1]
        br i1 %CN, label %T, label %F
T:              ; preds = %0
        call void @foo1( )
        ret void
F:              ; preds = %0
        call void @foo2( )
        ret void
}

define void @test3(i32 %V) {
        %C1 = icmp eq i32 %V, 4         ; <i1> [#uses=1]
        br i1 %C1, label %T, label %N
N:              ; preds = %0
        %C2 = icmp eq i32 %V, 17                ; <i1> [#uses=1]
        br i1 %C2, label %T, label %F
T:              ; preds = %N, %0
        call void @foo1( )
        ret void
F:              ; preds = %N
        call void @foo2( )
        ret void
}


