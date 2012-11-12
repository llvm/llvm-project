=============================
Code Reviews with Phabricator
=============================

.. contents::
  :local:

If you prefer to use a web user interface for code reviews,
you can now submit your patches for Clang and LLVM at
`LLVM's Phabricator`_.

Sign up
-------

There are two options to get an account on Phabricator. You can sign up
immediately with one of the supported OAuth account types if you're comfortable
with OAuth, but you can also email chandlerc@gmail.com to request an account to
be created manually without using OAuth. We're working to get support in
Phabricator to directly create new accounts, but currently this is a manual
process.

Note that if you use your Subversion user name as Phabricator user name,
Phabricator will automatically connect your submits to your Phabricator user in
the `Code Repository Browser`_.


Requesting a review via the command line
----------------------------------------

Phabricator has a tool called *Arcanist* to upload patches from
the command line. To get you set up, follow the
`Arcanist Quick Start`_ instructions.

You can learn more about how to use arc to interact with
Phabricator in the `Arcanist User Guide`_.

Requesting a review via the web interface
-----------------------------------------

The tool to create and review patches in Phabricator is called
*Differential*.

Note that you can upload patches created through various diff tools,
including git and svn. To make reviews easier, please always include
**as much context as possible** with your diff! Don't worry, Phabricator
will automatically send a diff with a smaller context in the review
email, but having the full file in the web interface will help the
reviewer understand your code.

To get a full diff, use one of the following commands (or just use Arcanist
to upload your patch):

* ``git diff -U999999 other-branch``
* ``svn diff --diff-cmd=diff -x -U999999``

To upload a new patch:

* Click *Differential*.
* Click *Create Revision*.
* Paste the text diff or upload the patch file.
  Note that TODO
* Leave the drop down on *Create a new Revision...* and click *Continue*.
* Enter a descriptive title and summary; add reviewers and mailing
  lists that you want to be included in the review. If your patch is
  for LLVM, cc llvm-commits; if your patch is for Clang, cc cfe-commits.
* Click *Save*.

To submit an updated patch:

* Click *Differential*.
* Click *Create Revision*.
* Paste the updated diff.
* Select the review you want to from the *Attach To* dropdown and click
  *Continue*.
* Click *Save*.

Reviewing code with Phabricator
-------------------------------

Phabricator allows you to add inline comments as well as overall comments
to a revision. To add an inline comment, select the lines of code you want
to comment on by clicking and dragging the line numbers in the diff pane.

You can add overall comments or submit your comments at the bottom of the page.

Phabricator has many useful features, for example allowing you to select
diffs between different versions of the patch as it was reviewed in the
*Revision Update History*. Most features are self descriptive - explore, and
if you have a question, drop by on #llvm in IRC to get help.

Status
------

Currently, we're testing Phabricator for use with Clang/LLVM. Please let us
know whether you like it and what could be improved!

.. _LLVM's Phabricator: http://llvm-reviews.chandlerc.com
.. _Code Repository Browser: http://llvm-reviews.chandlerc.com/diffusion/
.. _Arcanist Quick Start: http://www.phabricator.com/docs/phabricator/article/Arcanist_Quick_Start.html
.. _Arcanist User Guide: http://www.phabricator.com/docs/phabricator/article/Arcanist_User_Guide.html
