import unittest

import trailing_whitespace

class Test(unittest.TestCase):
    """Tests for trailing_whitespace."""

    def test_is_text(self) -> None:
        self.assertTrue(trailing_whitespace.is_text("foo.cpp"))
        self.assertTrue(trailing_whitespace.is_text("foo/bar.cpp"))
        self.assertFalse(trailing_whitespace.is_text("foo/bar"))
        self.assertFalse(trailing_whitespace.is_text("foo/bar.o"))

    def test_diff_parsing(self) -> None:
        basic_diff = """diff --git a/clang/test/CodeGen/memalign-libcall.c b/clang/test/CodeGen/memalign-libcall.c
index 2070eebdbf84..4fe1a838d15f 100644
--- a/clang/test/CodeGen/memalign-libcall.c
+++ b/clang/test/CodeGen/memalign-libcall.c
@@ -12 +12,2 @@ void *test(size_t alignment, size_t size) {
-// CHECK: attributes #2 = { nobuiltin "no-builtin-memalign" } 
\\ No newline at end of file
+// CHECK: attributes #2 = { nobuiltin "no-builtin-memalign" }"""
        self.assertEqual(trailing_whitespace.parse_diffs(basic_diff), {
            'clang/test/CodeGen/memalign-libcall.c': [
                (12, "// CHECK: attributes #2 = { nobuiltin \"no-builtin-memalign\" }")
            ]
        })
        multiple_added_lines = """diff --git a/clang/test/CodeGen/memalign-libcall.c b/clang/test/CodeGen/memalign-libcall.c
index 2070eebdbf84..4fe1a838d15f 100644
--- a/clang/test/CodeGen/memalign-libcall.c
+++ b/clang/test/CodeGen/memalign-libcall.c
@@ -12 +12,2 @@ void *test(size_t alignment, size_t size) {
-// CHECK: attributes #2 = { nobuiltin "no-builtin-memalign" } 
\\ No newline at end of file
+// CHECK: attributes #2 = { nobuiltin "no-builtin-memalign" }
+foobar"""
        self.assertEqual(trailing_whitespace.parse_diffs(multiple_added_lines), {
            'clang/test/CodeGen/memalign-libcall.c': [
                (12, "// CHECK: attributes #2 = { nobuiltin \"no-builtin-memalign\" }"),
                (13, "foobar"),
            ]
        })
        multiple_deleted = """diff --git a/clang/test/CodeGen/memalign-libcall.c b/clang/test/CodeGen/memalign-libcall.c
index 2070eebdbf84..4fe1a838d15f 100644
--- a/clang/test/CodeGen/memalign-libcall.c
+++ b/clang/test/CodeGen/memalign-libcall.c
@@ -12 +12,2 @@ void *test(size_t alignment, size_t size) {
-foobar
-// CHECK: attributes #2 = { nobuiltin "no-builtin-memalign" } 
\\ No newline at end of file
+// CHECK: attributes #2 = { nobuiltin "no-builtin-memalign" }"""
        self.assertEqual(trailing_whitespace.parse_diffs(multiple_deleted), {
            'clang/test/CodeGen/memalign-libcall.c': [
                (12, "// CHECK: attributes #2 = { nobuiltin \"no-builtin-memalign\" }"),
            ]
        })

if __name__ == "__main__":
    unittest.main()
