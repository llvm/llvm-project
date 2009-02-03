// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

enum TchkType {
  tchkNum, tchkString, tchkSCN, tchkNone
};

struct Operator {
  enum TchkType tchk[8];
};

struct Operator opTab[] = {
  {{tchkNum, tchkNum, tchkString} }
};

