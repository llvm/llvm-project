; RUN: opt -passes='simple-loop-unswitch<nontrivial>' %s -S | FileCheck %s

declare i1 @foo()

; CHECK: define {{.*}} @mem_cgroup_node_nr_lru_pages
define i32 @mem_cgroup_node_nr_lru_pages(i1 %tree) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end8, %entry
  br i1 %tree, label %if.end8, label %if.else

if.else:                                          ; preds = %for.cond
  callbr void asm sideeffect ".pushsection __jump_table,  \22aw\22 \0A\09.popsection \0A\09", "i,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@mem_cgroup_node_nr_lru_pages, %for.cond5))
          to label %if.end8 [label %for.cond5]

for.cond5:                                        ; preds = %if.else, %for.cond5
  %call6 = call i1 @foo()
  br i1 %call6, label %if.end8.loopexit, label %for.cond5

if.end8.loopexit:                                 ; preds = %for.cond5
  br label %if.end8

if.end8:                                          ; preds = %if.end8.loopexit, %if.else, %for.cond
  br label %for.cond
}

