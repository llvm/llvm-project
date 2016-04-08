add_lldb_unittest(SymbolFilePDBTests
  SymbolFilePDBTests.cpp
  )

set(test_inputs
   test-pdb.exe
   test-pdb.pdb
   test-dwarf.exe)

add_unittest_inputs(SymbolFilePDBTests "${test_inputs}")  
