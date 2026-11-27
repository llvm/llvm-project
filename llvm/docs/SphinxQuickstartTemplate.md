# Sphinx Quickstart Template

This article is intended to take someone in the state of “I want to write documentation and get it added to LLVM’s docs” and help them start writing documentation as fast as possible and with as little nonsense as possible.


## Overview

LLVM documentation is written in [Markedly Structured Text (MyST)][myst] and [reStructuredText (reST)][reStructuredText].
MyST is a Markdown flavor that adds Sphinx documentation extensions.
Markdown is preferred for new docs, and migrating old docs from reStructuredText to Markdown is an open, ongoing project.
[Sphinx], a documentation generator originally written for Python documentation, generates the LLVM HTML documentation from MyST and reST.

See the {ref}`migration <markdown_migration_guidelines>` section for more information on how to migrate existing docs.

[myst]: https://myst-parser.readthedocs.io/en/latest/
[reStructuredText]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[Sphinx]: http://www.sphinx-doc.org

## How to use this template

This article is located in `docs/SphinxQuickstartTemplate.md`.
To use it as a template, make a copy and open it in a text editor.
You can then write your docs, and open a [GitHub PR](GitHub.md) to request a review.

To view the Markdown source file for this article, click **Show Source** on the right sidebar.

## Authoring Guidelines

Focus on *content*. It is easy to fix the Markdown syntax later if necessary,
and Markdown intentionally imitates common plain-text conventions so it should
be quite natural. A basic knowledge of Markdown syntax is useful when writing
the document, so the last ~half of this document (starting with
[Example Section](#example-section)) gives examples which should cover 99% of
use cases.

Let me say that again: focus on *content*.
But if you really need to verify Sphinx's output, see `docs/README.txt` for information on how to build it.

### Creating New Articles

Before creating a new article, consider the following questions:

1. Why would I want to read this document?

1. What should I know to be able to follow along with this document?

1. What will I have learned by the end of this document?

A standard best practice is to make your articles task-oriented. You generally should not be writing documentation that isn't based around "how to" do something
unless there's already an existing "how to" article for the topic you're documenting. The reason for this is that without a "how to" article to read first, it might be difficult for
someone unfamiliar with the topic to understand a more advanced, conceptual article.

When creating a task-oriented article, follow existing LLVM articles by giving it a filename that starts with `HowTo*.md`. This format is usually the easiest for another person to understand and also the most useful.

Focus on content (yes, I had to say it again).

The rest of this document shows example Markdown and MyST markup constructs
that are meant to be read by you in your text editor after you have copied
this file into a new file for the documentation you are about to write.

## Example Section

An article can contain one or more sections (i.e., headings). Sections (like
`Example Section` above) help give your document its structure. Use `#` for the
document title, `##` for top-level sections, `###` for subsections, and so on.
Leave a blank line before and after each heading.

### Example Nested Subsection

Subsections can also be nested beneath other subsections. For more information
on Markdown syntax, see the [CommonMark spec] and the [MyST syntax guide].

[CommonMark spec]: https://spec.commonmark.org/
[MyST syntax guide]: https://myst-parser.readthedocs.io/en/latest/syntax/typography.html

## Text Formatting

Text can be *emphasized*, **bold**, or `monospace`.

To create a new paragraph, simply insert a blank line.

## Links

You can format a link [like this](https://llvm.org/). A more [sophisticated syntax] allows you to place the `[link text]: <URL>` block
pretty much anywhere else in the document. This is useful when linking to especially long URLs.

[sophisticated syntax]: http://en.wikipedia.org/wiki/LLVM

Use ordinary Markdown links for links within the same Sphinx documentation
project. Link to the source document, including its `.md` or `.rst` extension,
rather than to generated `.html` output. MyST resolves these as internal links
and the Sphinx build checks that the document exists:

```md
[LLVM's build documentation](GettingStarted.md)
```

To link to a heading, append its generated lowercase, hyphenated slug. MyST
also checks that the heading exists:

```md
[the CMake frequently-used variables](CMake.md#frequently-used-cmake-variables)
[an earlier section](#creating-new-articles)
```

Do not add the MyST `project:` prefix to source paths. A relative Markdown link
has the same Sphinx validation and also works when browsing the source on
GitHub. Likewise, do not link to generated paths such as `CMake.html`.

Use a Sphinx `{ref}` role when the target is an explicit label rather than a
generated heading. Explicit labels are useful when
an anchor must remain stable after its heading or source file is renamed.
Avoid adding explicit labels to ordinary headings when a checked Markdown link
is sufficient.

```md
(stable-section-label)=

## A heading whose text may change

See {ref}`the stable section <stable-section-label>`.
```

### Links to other LLVM projects

Do not link to another LLVM subproject's documentation with an absolute URL
such as `https://clang.llvm.org/docs/ThinLTO.html`. The Sphinx build cannot
check it, and it always points at the tip-of-tree site, even in a release or
downstream build of the documentation. Use an `inv:` link instead. It is
resolved through [intersphinx](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html)
against the other project's published inventory (`objects.inv`), and the build
warns if the target does not exist:

```md
[ThinLTO](inv:clang:std:doc#ThinLTO)
[the fmin/fmax family](inv:llvm:std:label#i_fminmax_family)
[`-fsave-optimization-record`](inv:clang:std:cmdoption#-f[no-]save-optimization-record)
```

The link target has the form `inv:<project>:<domain>:<type>#<target>`:

* `<project>` is the key in the project's `intersphinx_mapping`: `llvm`,
  `clang`, `clang-tools-extra`, `lld`, `lldb`, `flang`, `polly`, `libc`,
  `libcxx`, or `openmp`. Projects take these from `LLVM_INTERSPHINX_MAPPING` in
  `utils/docs/llvm_sphinx/__init__.py`.
* `<domain>` is the Sphinx domain. Documents, labels, and command-line options
  all live in the standard domain, `std`.
* `<type>` is the kind of object. Use `doc` for a whole document, `label` for an
  explicit label, and `cmdoption` for a command-line option.
* `<target>` is the name within that type. For `doc`, it is the document name
  relative to the other project's source root, without an extension (for example
  `BoundsSafety/BoundsSafetyAdoptionGuide`). Labels are stored in lowercase.

If the link text is empty, as in `[](inv:clang:std:doc#ThinLTO)`, the title from
the inventory is used. Prefer writing the text explicitly so the source reads
naturally. In reStructuredText, use the equivalent role,
``:external+clang:doc:`ThinLTO```.

To find the type and target name for a page or label, list the project's
inventory:

```sh
python -m sphinx.ext.intersphinx https://clang.llvm.org/docs/objects.inv
```

`inv:` links only resolve in the Sphinx build, so a document that is read outside
it, for example a file in the repository root, should link to it on GitHub
instead. A project must set `intersphinx_mapping` in its `conf.py`, for example
`intersphinx_mapping = LLVM_INTERSPHINX_MAPPING.copy()`, before it can use these
links.

### How the inventories are found

By default, a documentation build downloads the other projects' published
inventories. That is simple, but it needs network access, and a link to a label
that was added by the same change does not resolve until the change is
published. Unresolved links are reported as warnings, and the rendered page has
no link.

CI builds are hermetic instead: every inventory comes from the sources in the
checkout, so all links resolve, including circular ones between projects. The
`utils/docs/hermetic_build` CMake project builds the docs of all of the
projects this way:

```sh
cmake -G Ninja -S utils/docs/hermetic_build -B build-docs
ninja -C build-docs docs
```

It builds the tools (`LLVM_ENABLE_PROJECTS`) and the runtimes
(`LLVM_ENABLE_RUNTIMES`), which are separate CMake builds, in parallel. First it
builds each project's inventory, which only reads its sources, with the
`inventory` builder in `llvm_sphinx.ext.inventory` (target `docs-inventory`).
Then it builds the docs, which read all of the inventories. See
`LLVM_SPHINX_INVENTORY_DIR` in [](CMake.md) for doing this in your own build.

## Lists

Markdown allows you to create ordered lists...

1. A list starting with `1.` will be automatically numbered.

1. This is a second list element.

   1. Use indentation to create nested lists.

...as well as unordered lists:

* Stuff.

  + Deeper stuff.

* More stuff.

## Code Blocks

You can make blocks of code like this:

```cpp
int main() {
  return 0;
}
```

For a shell session, use a `console` code block (some existing docs use
`bash`):

```console
$ echo "Goodbye cruel world!"
$ rm -rf /
```

If you need to show LLVM IR, use the `llvm` code block.

```llvm
define i32 @test1() {
entry:
  ret i32 0
}
```

Some other common code blocks you might need are `c`, `objc`, `make`,
and `cmake`. If you need something beyond that, you can look at the [full
list] of supported code blocks.

[full list]: http://pygments.org/docs/lexers/

However, don't waste time fiddling with syntax highlighting when you could
be adding meaningful content. When in doubt, show preformatted text
without any syntax highlighting like this:

```
                          .
                           +:.
                       ..:: ::
                    .++:+:: ::+:.:.
                   .:+           :
            ::.::..::            .+.
          ..:+    ::              :
    ......+:.                    ..
          :++.    ..              :
            .+:::+::              :
            ..   . .+            ::
                     +.:      .::+.
                      ...+. .: .
                         .++:..
                          ...
```


## Sphinx Directives

Sphinx directives that contain rich, formatted Markdown content should use
[colon fences][colon_fence] instead of backtick fences. This keeps directive
containers visually distinct from code blocks while still allowing normal
Markdown markup inside the directive.

:::{note}
Use colon fences for rich directives such as `{note}`, `{warning}`, `{option}`,
and similar content blocks.
:::

Prefer triple backtick fences for code regions and literal examples:

```cpp
int value = 0;
```

Keep `{eval-rst}` blocks fenced with backticks, because their contents are
reStructuredText rather than Markdown.

[colon_fence]: https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#syntax-colon-fence

## Generating the documentation

You can generate the HTML documentation from the sources locally if you want to
see what they would look like. In addition to the normal
[build tools](GettingStarted.md)
you need to install [Sphinx] and the necessary extensions
using the following command inside the `llvm-project` checkout:

```console
pip install --user -r ./llvm/docs/requirements.txt
```

Then run cmake to build the documentation inside the `llvm-project` checkout:

```console
mkdir build
cd build
cmake -DLLVM_ENABLE_SPHINX=On ../llvm
cmake --build . --target docs-llvm-html
```

In case you already have the Cmake build set up and want to reuse that,
just set the CMake variable `LLVM_ENABLE_SPHINX=On`.

After that you find the generated documentation in `build/docs/html`
folder.

(markdown_migration_guidelines)=

## Markdown migration guidelines

These are some goals to keep in mind during a migration:

* Optimize for review: Decompose the migration into mechanical steps that are easy to review and validate.
* Enable source code archaelogy: Expect future readers to want to know when documentation policy changed, so keep your changes mechanical to make that easy.
* Minimize conflicts: LLVM is a big community of many developers with lots of development branches and downstreams. Make conflict resolution easy.

For that reason, it's helpful to create 2-3 {ref}`stacked pull requests <stacked_pull_requests>`:

* Rename `.rst` -> `.md` and update cross-references. This will presumably break the docs build, but to follow the one-PR-per-commit policy, it must be its own PR.
* Mechanically update reST conventions to markdown (\`\` -> \`). Avoid unnecessary reflow. This means you should avoid tools like pandoc, which reflow paragraphs.
* (optional) Reflow text in line with current project policy. You don't have to do this, but if you feel the need to, make it a separate step. This can be resequenced as step 1. Separating out the text reflow step makes it easy to review and follow blame.

Don't rely too heavily on automated error checking to catch any documentation bugs.
If you are migrating a long doc, you are reponsible for building the docs locally and validating the rendering yourself using the steps above.
