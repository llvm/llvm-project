# How to Update Unicode Tables

LLVM and Clang use a subset of the [Unicode Character Database](https://www.unicode.org/ucd/)
(UCD) for identifiers, `\N{...}` named characters, diagnostic column width,
and simple case folding. The corresponding tables are generated from the UCD
into Clang and LLVM Support.

## Downloading the Unicode data

Download the following files, replacing `<VERSION>` with a Unicode version such
as `18.0.0`:

```text
https://www.unicode.org/Public/<VERSION>/ucdxml/ucd.nounihan.flat.zip
https://www.unicode.org/Public/<VERSION>/ucd/UnicodeData.txt
https://www.unicode.org/Public/<VERSION>/ucd/NameAliases.txt
https://www.unicode.org/Public/<VERSION>/ucd/extracted/DerivedName.txt
```

Unzip `ucd.nounihan.flat.zip` to get `ucd.nounihan.flat.xml`.

## Building the generators

Build the generators from an LLVM build directory that has utilities enabled
(`LLVM_BUILD_UTILS`, the default) and libxml2 (`LLVM_ENABLE_LIBXML2`, also the
default):

```bash
ninja -C <build> UnicodeCharSetsGenerator UnicodeNameMappingGenerator
```

Then run them from the `llvm-project` root as shown below.

## Character properties

`UnicodeCharSetsGenerator` writes the identifier character sets used by the
lexer (`XID_Start`, `XID_Continue`, and the mathematical compatibility notation
profile) and the printable / formatting / combining / East-Asian-width sets
used by LLVM Support.

```bash
<build>/bin/UnicodeCharSetsGenerator ucd.nounihan.flat.xml \
    clang/lib/Lex/UnicodeCharSetsGenerated.cpp \
    llvm/lib/Support/UnicodeCharSetsGenerated.cpp
clang-format -i clang/lib/Lex/UnicodeCharSetsGenerated.cpp \
    llvm/lib/Support/UnicodeCharSetsGenerated.cpp
```

The C99 and C11 identifier tables and the whitespace table in
`clang/lib/Lex/UnicodeCharSets.h` are maintained by hand.

## Character names

`UnicodeNameMappingGenerator` writes the name-to-codepoint trie used by
`\N{...}` and `llvm::sys::unicode::nameToCodepointStrict` /
`nameToCodepointLooseMatching`.

```bash
<build>/bin/UnicodeNameMappingGenerator UnicodeData.txt NameAliases.txt \
    llvm/lib/Support/UnicodeNameToCodepointGenerated.cpp
clang-format -i llvm/lib/Support/UnicodeNameToCodepointGenerated.cpp
```

Algorithmically derived names (`CJK UNIFIED IDEOGRAPH-*`, and so on) are not
in those files. Update `GeneratedNamesDataTable` in
`llvm/lib/Support/UnicodeNameToCodepoint.cpp` from `DerivedName.txt`.

## Case folding

`llvm/utils/unicode-case-fold.py` fetches `CaseFolding.txt` and writes
`llvm/lib/Support/UnicodeCaseFold.cpp`:

```bash
llvm/utils/unicode-case-fold.py \
    https://www.unicode.org/Public/<VERSION>/ucd/CaseFolding.txt \
    > llvm/lib/Support/UnicodeCaseFold.cpp
```

After regenerating the tables, update
`llvm/unittests/Support/UnicodeTest.cpp` and `clang/test/Lexer/unicode.c` for
new characters and changed name ranges.
