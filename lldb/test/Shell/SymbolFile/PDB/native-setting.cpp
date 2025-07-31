// REQUIRES: target-windows

// Test plugin.symbol-file.pdb.use-native-reader setting
// RUN: %build -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=0 %lldb %t.exe -o 'target modules dump symfile' | FileCheck --check-prefix=ENV0 %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb %t.exe -o 'target modules dump symfile' | FileCheck --check-prefix=ENV1 %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=0 %lldb \
// RUN:     -o 'settings set plugin.symbol-file.pdb.use-native-reader off' \
// RUN:     -o 'target create %t.exe' \
// RUN:     -o 'target modules dump symfile' \
// RUN:     | FileCheck --check-prefix=ENV0-SET0 %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb \
// RUN:     -o 'settings set plugin.symbol-file.pdb.use-native-reader off' \
// RUN:     -o 'target create %t.exe' \
// RUN:     -o 'target modules dump symfile' \
// RUN:     | FileCheck --check-prefix=ENV1-SET0 %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=0 %lldb \
// RUN:     -o 'settings set plugin.symbol-file.pdb.use-native-reader on' \
// RUN:     -o 'target create %t.exe' \
// RUN:     -o 'target modules dump symfile' \
// RUN:     | FileCheck --check-prefix=ENV0-SET1 %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb \
// RUN:     -o 'settings set plugin.symbol-file.pdb.use-native-reader on' \
// RUN:     -o 'target create %t.exe' \
// RUN:     -o 'target modules dump symfile' \
// RUN:     | FileCheck --check-prefix=ENV1-SET1 %s

// ENV0: (lldb) target modules dump symfile
// ENV0: Dumping debug symbols for 1 modules.
// ENV0: SymbolFile pdb

// ENV1: (lldb) target modules dump symfile
// ENV1: Dumping debug symbols for 1 modules.
// ENV1: SymbolFile native-pdb

// ENV0-SET0: (lldb) target modules dump symfile
// ENV0-SET0: Dumping debug symbols for 1 modules.
// ENV0-SET0: SymbolFile pdb

// ENV1-SET0: (lldb) target modules dump symfile
// ENV1-SET0: Dumping debug symbols for 1 modules.
// ENV1-SET0: SymbolFile pdb

// ENV0-SET1: (lldb) target modules dump symfile
// ENV0-SET1: Dumping debug symbols for 1 modules.
// ENV0-SET1: SymbolFile native-pdb

// ENV1-SET1: (lldb) target modules dump symfile
// ENV1-SET1: Dumping debug symbols for 1 modules.
// ENV1-SET1: SymbolFile native-pdb

int main() {}
