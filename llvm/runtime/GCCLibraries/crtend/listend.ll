; global_ctors/global_dtors terminator: this is used to add a terminating null
; value to the initialization list.

%struct..TorRec = type { int, void ()* }

%llvm.global_ctors = appending global [1 x %struct..TorRec] [
    %struct..TorRec { int 2147483647, void ()* null }
  ]

%llvm.global_dtors = appending global [1 x %struct..TorRec] [
    %struct..TorRec { int 2147483647, void ()* null }
  ]

implementation

%struct..TorRec* %__llvm_getGlobalCtors() {
  ret %struct..TorRec* getelementptr ([1 x %struct..TorRec]* %llvm.global_ctors,
                                     long 0, long 0)
}
%struct..TorRec* %__llvm_getGlobalDtors() {
  ret %struct..TorRec* getelementptr ([1 x %struct..TorRec]* %llvm.global_dtors,
                                     long 0, long 0)
}
