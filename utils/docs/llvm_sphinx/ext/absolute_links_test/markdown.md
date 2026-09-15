# Markdown Absolute Link Tests

This [absolute link](http://www.example.test/docs/target.html?view=1#target-section)
points into this Sphinx project.

These nonportable internal links should warn:

- [explicit project document](project:target.md)
- [explicit project section](project:target.md#target-section)
- [generated HTML document](target.html)
- [generated HTML section](target.html#target-section)
- [reference-style project link][project-target]
- [reference-style generated HTML link][html-target]

[project-target]: project:target.md#target-document
[html-target]: target.html#target-document

:::{note}
[A project link in Markdown directive content](project:rest.rst)
:::

These links should not warn:

- [another project](https://other.example.test/docs/target.html)
- [a nonexistent document](https://example.test/docs/missing.html)
- [a non-document page](https://example.test/docs/downloads/package.tar.xz)
- [source document](target.md)
- [reStructuredText source document](rest.rst)
- [source heading](target.md#target-section)
- [same-document heading](#markdown-absolute-link-tests)
- [an HTML file that is not a document](static.html)

```md
[a project link in an example](project:target.md)
```
