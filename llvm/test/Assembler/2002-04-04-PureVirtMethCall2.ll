; RUN: llvm-as %s -o /dev/null -f

%t = type { { \2*, \2 },
            { \2*, \2 }
          }
