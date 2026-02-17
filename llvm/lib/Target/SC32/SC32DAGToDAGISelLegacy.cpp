#include "SC32DAGToDAGISelLegacy.h"
#include "SC32DAGToDAGISel.h"

using namespace llvm;

char SC32DAGToDAGISelLegacy::ID = 0;

SC32DAGToDAGISelLegacy::SC32DAGToDAGISelLegacy(TargetMachine &TM)
    : SelectionDAGISelLegacy(ID, std::make_unique<SC32DAGToDAGISel>(TM)) {}
