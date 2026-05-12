// REQUIRES: !diasdk, target-windows

// Test plugin.symbol-file.pdb.reader setting without the DIA SDK
// RUN: %build -o %t.exe -- %s
// RUN: env -u LLDB_USE_NATIVE_PDB_READER %lldb %t.exe -o 'target modules dump symfile' 2>&1 | FileCheck --check-prefix=NO-ENV %s
// RUN: env LLDB_USE_NATIVE_PDB_READER= %lldb %t.exe -o 'target modules dump symfile' 2>&1 | FileCheck --check-prefix=NO-ENV %s

// RUN: env LLDB_USE_NATIVE_PDB_READER=0 %lldb %t.exe -o 'target modules dump symfile' 2>&1 | FileCheck --check-prefix=ENV0 %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb %t.exe -o 'target modules dump symfile' 2>&1 | FileCheck --check-prefix=ENV1 %s

// RUN: env LLDB_USE_NATIVE_PDB_READER=foo %lldb %t.exe -o 'target modules dump symfile' 2>&1 | FileCheck --check-prefix=ENV1 %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=42 %lldb %t.exe -o 'target modules dump symfile' 2>&1 | FileCheck --check-prefix=ENV1 %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=-1 %lldb %t.exe -o 'target modules dump symfile' 2>&1 | FileCheck --check-prefix=ENV1 %s

// RUN: env LLDB_USE_NATIVE_PDB_READER=0 %lldb \
// RUN:     -o 'settings set plugin.symbol-file.pdb.reader dia' \
// RUN:     -o 'target create %t.exe' \
// RUN:     -o 'target modules dump symfile' \
// RUN:     2>&1 | FileCheck --check-prefix=ENV0-SET-DIA %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb \
// RUN:     -o 'settings set plugin.symbol-file.pdb.reader dia' \
// RUN:     -o 'target create %t.exe' \
// RUN:     -o 'target modules dump symfile' \
// RUN:     2>&1 | FileCheck --check-prefix=ENV1-SET-DIA %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=0 %lldb \
// RUN:     -o 'settings set plugin.symbol-file.pdb.reader native' \
// RUN:     -o 'target create %t.exe' \
// RUN:     -o 'target modules dump symfile' \
// RUN:     2>&1 | FileCheck --check-prefix=ENV0-SET-NATIVE %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb \
// RUN:     -o 'settings set plugin.symbol-file.pdb.reader native' \
// RUN:     -o 'target create %t.exe' \
// RUN:     -o 'target modules dump symfile' \
// RUN:     2>&1 | FileCheck --check-prefix=ENV1-SET-NATIVE %s

// NO-ENV-NOT: warning:
// NO-ENV: (lldb) target modules dump symfile
// NO-ENV: Dumping debug symbols for 1 modules.
// NO-ENV: SymbolFile native-pdb

// ENV0: warning: the DIA PDB reader was explicitly requested, but LLDB was built without the DIA SDK. The native reader will be used instead
// ENV0: (lldb) target modules dump symfile
// ENV0: Dumping debug symbols for 1 modules.
// ENV0: SymbolFile native-pdb

// ENV1-NOT: warning:
// ENV1: (lldb) target modules dump symfile
// ENV1: Dumping debug symbols for 1 modules.
// ENV1: SymbolFile native-pdb

// ENV0-SET-DIA: warning: the DIA PDB reader was explicitly requested, but LLDB was built without the DIA SDK. The native reader will be used instead
// ENV0-SET-DIA: (lldb) target modules dump symfile
// ENV0-SET-DIA: Dumping debug symbols for 1 modules.
// ENV0-SET-DIA: SymbolFile native-pdb

// ENV1-SET-DIA: warning: the DIA PDB reader was explicitly requested, but LLDB was built without the DIA SDK. The native reader will be used instead
// ENV1-SET-DIA: (lldb) target modules dump symfile
// ENV1-SET-DIA: Dumping debug symbols for 1 modules.
// ENV1-SET-DIA: SymbolFile native-pdb

// ENV1-SET-NATIVE-NOT: warning:
// ENV0-SET-NATIVE: (lldb) target modules dump symfile
// ENV0-SET-NATIVE: Dumping debug symbols for 1 modules.
// ENV0-SET-NATIVE: SymbolFile native-pdb

// ENV1-SET-NATIVE-NOT: warning:
// ENV1-SET-NATIVE: (lldb) target modules dump symfile
// ENV1-SET-NATIVE: Dumping debug symbols for 1 modules.
// ENV1-SET-NATIVE: SymbolFile native-pdb

int main() {}
