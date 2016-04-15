add_lldb_unittest(SymbolFilePDBTests
  SymbolFilePDBTests.cpp
  )

set(test_inputs
   test-pdb.exe
   test-pdb.pdb
   test-dwarf.exe
   test-pdb-types.exe
   test-pdb-types.pdb)

add_unittest_inputs(SymbolFilePDBTests "${test_inputs}")  
