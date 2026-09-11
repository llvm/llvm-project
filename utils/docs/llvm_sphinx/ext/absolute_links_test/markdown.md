# Markdown Absolute Link Tests

This [absolute link](http://www.example.test/docs/target.html?view=1#target-section)
points into this Sphinx project.

This [cross-project link](https://other.example.test/docs/target.html#external-section)
points into a configured intersphinx project.
Its [project root](https://other.example.test/docs) does too.

These links should not warn:

- [an intersphinx inventory link](inv:other:std:doc#target)
- [an unconfigured project](https://unconfigured.example.test/docs/target.html)
- [a nonexistent document](https://example.test/docs/missing.html)
- [a non-document page](https://example.test/docs/downloads/package.tar.xz)
