==========================
Sphinx Quickstart Template
==========================

.. sectionauthor:: Sean Silva <silvas@purdue.edu>

Introduction and Quickstart
===========================

This document is meant to get you writing documentation as fast as possible
even if you have no previous experience with Sphinx. The goal is to take
someone in the state of "I want to write documentation and get it added to
LLVM's docs" and turn that into useful documentation mailed to llvm-commits
with as little nonsense as possible.

You can find this document in ``docs/SphinxQuickstartTemplate.rst``. You
should copy it, open the new file in your text editor, write your docs, and
then send the new document to llvm-commits for review.

Focus on *content*. It is easy to fix the Sphinx (reStructuredText) syntax
later if necessary, although reStructuredText tries to imitate common
plain-text conventions so it should be quite natural. A basic knowledge of
reStructuredText syntax is useful when writing the document, so the last
~half of this document (starting with `Example Section`_) gives examples
which should cover 99% of use cases.

Let me say that again: focus on *content*.

Once you have finished with the content, please send the ``.rst`` file to
llvm-commits for review.

Guidelines
==========

Try to answer the following questions in your first section:

#. Why would I want to read this document?

#. What should I know to be able to follow along with this document?

#. What will I have learned by the end of this document?

Common names for the first section are ``Introduction``, ``Overview``, or
``Background``.

If possible, make your document a "how to". Give it a name ``HowTo*.rst``
like the other "how to" documents. This format is usually the easiest
for another person to understand and also the most useful.

You generally should not be writing documentation other than a "how to"
unless there is already a "how to" about your topic. The reason for this
is that without a "how to" document to read first, it is difficult for a
person to understand a more advanced document.

Focus on content (yes, I had to say it again).

The rest of this document shows example reStructuredText markup constructs
that are meant to be read by you in your text editor after you have copied
this file into a new file for the documentation you are about to write.

Example Section
===============

Your text can be *emphasized*, **bold**, or ``monospace``.

Use blank lines to separate paragraphs.

Headings (like ``Example Section`` just above) give your document
structure. Use the same kind of adornments (e.g. ``======`` vs. ``------``)
as are used in this document. The adornment must be the same length as the
text above it. For Vim users, variations of ``yypVr=`` might be handy.

Example Subsection
------------------

Make a link `like this <http://llvm.org/>`_. There is also a more
sophisticated syntax which `can be more readable`_ for longer links since
it disrupts the flow less. You can put the ``.. _`link text`: <URL>`` block
pretty much anywhere later in the document.

.. _`can be more readable`: http://en.wikipedia.org/wiki/LLVM

Lists can be made like this:

#. A list starting with ``#.`` will be automatically numbered.

#. This is a second list element.

   #. They nest too.

You can also use unordered lists.

* Stuff.

  + Deeper stuff.

* More stuff.

Example Subsubsection
^^^^^^^^^^^^^^^^^^^^^

You can make blocks of code like this:

.. code-block:: c++

   int main() {
     return 0
   }

For a shell session, use a ``bash`` code block:

.. code-block:: bash

   $ echo "Goodbye cruel world!"
   $ rm -rf /

If you need to show LLVM IR use the ``llvm`` code block.

Hopefully you won't need to be this deep
""""""""""""""""""""""""""""""""""""""""

If you need to do fancier things than what has been shown in this document,
you can mail the list or check Sphinx's `reStructuredText Primer`_.

.. _`reStructuredText Primer`: http://sphinx.pocoo.org/rest.html
