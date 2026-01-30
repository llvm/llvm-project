# RUN: env PYTHONPATH=%llvm_lib_dir %python %s | FileCheck %s

import ir2vec


def test_invalid_file():
    """Test that invalid file path raises ValueError"""
    try:
        tool = ir2vec.initEmbedding(
            filename="/this/does/not/exist.ll",
            mode="sym",
            vocabPath="/also/fake/vocab.json",
        )
        return "FAIL: No exception raised"
    except ValueError as e:
        return f"PASS: ValueError raised - {str(e)[:40]}"
    except Exception as e:
        return f"FAIL: Wrong exception - {type(e).__name__}"


def test_empty_filename():
    """Test that empty filename raises ValueError"""
    try:
        tool = ir2vec.initEmbedding(filename="", mode="sym", vocabPath="dummy.json")
        return "FAIL: No exception raised"
    except ValueError:
        return "PASS: ValueError raised for empty filename"
    except Exception as e:
        return f"FAIL: Wrong exception - {type(e).__name__}"


result1 = test_invalid_file()
print(f"Test 1: {result1}")
# CHECK: Test 1: PASS: ValueError raised

result2 = test_empty_filename()
print(f"Test 2: {result2}")
# CHECK: Test 2: PASS: ValueError raised
