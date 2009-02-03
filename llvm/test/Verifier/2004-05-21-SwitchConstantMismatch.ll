; RUN: not llvm-as < %s >& /dev/null



int %main() {  
start1:
  switch uint 0, label %brt0 [int 3, label %brt1  ]
brt0:
  ret int 0
brt1:
  ret int 0
}

