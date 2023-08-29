==============
MyFirstTypoFix
==============

.. contents::
   :local:

Introduction
============

This tutorial will guide you through the process of making a change to
LLVM, and contributing it back to the LLVM project. We'll be making a
change to Clang, but the steps for other parts of LLVM are the same.
Even though the change we'll be making is simple, we're going to cover
steps like building LLVM, running the tests, and code review. This is
good practice, and you'll be prepared for making larger changes.

We'll assume you:

-  know how to use an editor,

-  have basic C++ knowledge,

-  know how to install software on your system,

-  are comfortable with the command line,

-  have basic knowledge of git.


The change we're making
-----------------------

Clang has a warning for infinite recursion:

.. code:: console

   $ echo "void foo() { foo(); }" > ~/test.cc
   $ clang -c -Wall ~/test.cc
   input.cc:1:14: warning: all paths through this function will call
   itself [-Winfinite-recursion]

This is clear enough, but not exactly catchy. Let's improve the wording
a little:

.. code:: console

   input.cc:1:14: warning: to understand recursion, you must first
   understand recursion [-Winfinite-recursion]


Dependencies
------------

We're going to need some tools:

-  git: to check out the LLVM source code,

-  a C++ compiler: to compile LLVM source code. You'll want `a recent
   version <https://llvm.org/docs/GettingStarted.html#host-c-toolchain-both-compiler-and-standard-library>`__
   of Clang, GCC, or Visual Studio.

-  CMake: used to configure how LLVM should be built on your system,

-  ninja: runs the C++ compiler to (re)build specific parts of LLVM,

-  python: to run the LLVM tests,

-  arcanist: for uploading changes for review,

As an example, on Ubuntu:

.. code:: console

   $ sudo apt-get install git clang cmake ninja-build python arcanist


Building LLVM
=============


Checkout
--------

The source code is stored `on
Github <https://github.com/llvm/llvm-project>`__ in one large repository
("the monorepo").

It may take a while to download!

.. code:: console

   $ git clone https://github.com/llvm/llvm-project.git

This will create a directory "llvm-project" with all of the source
code.(Checking out anonymously is OK - pushing commits uses a different
mechanism, as we'll see later)

Configure your workspace
------------------------

Before we can build the code, we must configure exactly how to build it
by running CMake. CMake combines information from three sources:

-  explicit choices you make (is this a debug build?)

-  settings detected from your system (where are libraries installed?)

-  project structure (which files are part of 'clang'?)

First, create a directory to build in. Usually, this is
llvm-project/build.

.. code:: console

   $ mkdir llvm-project/build
   $ cd llvm-project/build

Now, run CMake:

.. code:: console

   $ cmake -G Ninja ../llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=clang

If all goes well, you'll see a lot of "performing test" lines, and
finally:

.. code:: console

   Configuring done
   Generating done
   Build files have been written to: /path/llvm-project/build

And you should see a build.ninja file.

Let's break down that last command a little:

-  **-G Ninja**: we're going to use ninja to build; please create
   build.ninja

-  **../llvm**: this is the path to the source of the "main" LLVM
   project

-  The two **-D** flags set CMake variables, which override
   CMake/project defaults:

-  **CMAKE\ BUILD\ TYPE=Release**: build in optimized mode, which is
   (surprisingly) the fastest option.

   If you want to run under a debugger, you should use the default Debug
   (which is totally unoptimized, and will lead to >10x slower test
   runs) or RelWithDebInfo which is a halfway point.
   **CMAKE\ BUILD\ TYPE** affects code generation only, assertions are
   on by default regardless! **LLVM\ ENABLE\ ASSERTIONS=Off** disables
   them.

-  **LLVM\ ENABLE\ PROJECTS=clang** : this lists the LLVM subprojects
   you are interested in building, in addition to LLVM itself. Multiple
   projects can be listed, separated by semicolons, such as "clang;
   lldb".In this example, we'll be making a change to Clang, so we
   should build it.

Finally, create a symlink (or a copy) of
llvm-project/build/compile-commands.json into llvm-project/:

.. code:: console

   $ ln -s build/compile_commands.json ../

(This isn't strictly necessary for building and testing, but allows
tools like clang-tidy, clang-query, and clangd to work in your source
tree).


Build and test
--------------

Finally, we can build the code! It's important to do this first, to
ensure we're in a good state before making changes. But what to build?
In ninja, you specify a **target**. If we just want to build the clang
binary, our target name is "clang" and we run:

.. code:: console

   $ ninja clang

The first time we build will be very slow - Clang + LLVM is a lot of
code. But incremental builds are fast: ninja will only rebuild the parts
that have changed. When it finally finishes you should have a working
clang binary. Try running:

.. code:: console

   $ bin/clang --version

There's also a target for building and running all the clang tests:

.. code:: console

   $ ninja check-clang

This is a common pattern in LLVM: check-llvm is all the checks for core,
other projects have targets like check-lldb.


Making changes
==============


Edit
----

We need to find the file containing the error message.

.. code:: console

   $ git grep "all paths through this function" ..
   ../clang/include/clang/Basic/DiagnosticSemaKinds.td:  "all paths through this function will call itself">,

The string that appears in DiagnosticSemaKinds.td is the one that is
printed by Clang. \*.td files define tables - in this case it's a list
of warnings and errors clang can emit and their messages. Let's update
the message in your favorite editor:

.. code:: console

   $ vi ../clang/include/clang/Basic/DiagnosticSemaKinds.td

Find the message (it should be under
warn\ *infinite*\ recursive_function)Change the message to "in order to
understand recursion, you must first understand recursion".


Test again
----------

To verify our change, we can build clang and manually check that it
works.

.. code:: console

   $ ninja clang
   $ bin/clang -Wall ~/test.cc
   /path/test.cc:1:124: warning: in order to understand recursion, you must
   first understand recursion [-Winfinite-recursion]

We should also run the tests to make sure we didn't break something.

.. code:: console

   $ ninja check-clang

Notice that it is much faster to build this time, but the tests take
just as long to run. Ninja doesn't know which tests might be affected,
so it runs them all.

.. code:: console

   ********************
   Testing Time: 408.84s
   ********************
   Failing Tests (1):
       Clang :: SemaCXX/warn-infinite-recursion.cpp

Well, that makes sense… and the test output suggests it's looking for
the old string "call itself" and finding our new message instead.
Note that more tests may fail in a similar way as new tests are
added time to time.

Let's fix it by updating the expectation in the test.

.. code:: console

   $ vi ../clang/test/SemaCXX/warn-infinite-recursion.cpp

Everywhere we see `// expected-warning{{call itself}}` (or something similar
from the original warning text), let's replace it with
`// expected-warning{{to understand recursion}}`.

Now we could run **all** the tests again, but this is a slow way to
iterate on a change! Instead, let's find a way to re-run just the
specific test. There are two main types of tests in LLVM:

-  **lit tests** (e.g. SemaCXX/warn-infinite-recursion.cpp).

These are fancy shell scripts that run command-line tools and verify the
output. They live in files like
clang/**test**/FixIt/dereference-addressof.c. Re-run like this:

.. code:: console

   $ bin/llvm-lit -v ../clang/test/SemaCXX/warn-infinite-recursion.cpp

-  **unit tests** (e.g. ToolingTests/ReplacementTest.CanDeleteAllText)

These are C++ programs that call LLVM functions and verify the results.
They live in suites like ToolingTests. Re-run like this:

.. code:: console

   $ ninja ToolingTests && tools/clang/unittests/Tooling/ToolingTests
   --gtest_filter=ReplacementTest.CanDeleteAllText


Commit locally
--------------

We'll save the change to a local git branch. This lets us work on other
things while the change is being reviewed. Changes should have a
description, to explain to reviewers and future readers of the code why
the change was made.

.. code:: console

   $ git checkout -b myfirstpatch
   $ git commit -am "[Diagnostic] Clarify -Winfinite-recursion message"

Now we're ready to send this change out into the world! By the way,
There is an unwritten convention of using tag for your commit. Tags
usually represent modules that you intend to modify. If you don't know
the tags for your modules, you can look at the commit history :
https://github.com/llvm/llvm-project/commits/main.


Code review
===========


Finding a reviewer
------------------

Changes can be reviewed by anyone in the LLVM community who has commit
access.For larger and more complicated changes, it's important that the
reviewer has experience with the area of LLVM and knows the design goals
well. The author of a change will often assign a specific reviewer (git
blame and git log can be useful to find one).

As our change is fairly simple, we'll add the cfe-commits mailing list
as a subscriber; anyone who works on clang can likely pick up the
review. (For changes outside clang, llvm-commits is the usual list. See
`http://lists.llvm.org/ <http://lists.llvm.org/mailman/listinfo>`__ for
all the \*-commits mailing lists).


Uploading a change for review
-----------------------------

LLVM code reviews happen at https://reviews.llvm.org. The web interface
is called Phabricator, and the code review part is Differential. You
should create a user account there for reviews (click "Log In" and then
"Register new account").

Now you can upload your change for review:

.. code:: console

   $ arc diff HEAD^

This creates a review for your change, comparing your current commit
with the previous commit. You will be prompted to fill in the review
details. Your commit message is already there, so just add cfe-commits
under the "subscribers" section. It should print a code review URL:
https://reviews.llvm.org/D58291 You can always find your active reviews
on Phabricator under "My activity".


Review process
--------------

When you upload a change for review, an email is sent to you, the
cfe-commits list, and anyone else subscribed to these kinds of changes.
Within a few days, someone should start the review. They may add
themselves as a reviewer, or simply start leaving comments. You'll get
another email any time the review is updated. The details are in the
`https://llvm.org/docs/CodeReview/ <https://llvm.org/docs/CodeReview.html>`__.


Comments
~~~~~~~~

The reviewer can leave comments on the change, and you can reply. Some
comments are attached to specific lines, and appear interleaved with the
code. You can either reply to these, or address them and mark them as
"done". Note that in-line replies are **not** sent straight away! They
become "draft" comments and you must click "Submit" at the bottom of the
page.


Updating your change
~~~~~~~~~~~~~~~~~~~~

If you make changes in response to a reviewer's comments, simply run

.. code:: console

   $ arc diff

again to update the change and notify the reviewer. Typically this is a
good time to send any draft comments as well.


Accepting a revision
~~~~~~~~~~~~~~~~~~~~

When the reviewer is happy with the change, they will **Accept** the
revision. They may leave some more minor comments that you should
address, but at this point the review is complete. It's time to get it
committed!


Commit by proxy
---------------

As this is your first change, you won't have access to commit it
yourself yet. The reviewer **doesn't know this**, so you need to tell
them! Leave a message on the review like:

   Thanks @somellvmdev. I don't have commit access, can you land this
   patch for me? Please use "My Name my@email" to commit the change.

The review will be updated when the change is committed.


Review expectations
-------------------

In order to make LLVM a long-term sustainable effort, code needs to be
maintainable and well tested. Code reviews help to achieve that goal.
Especially for new contributors, that often means many rounds of reviews
and push-back on design decisions that do not fit well within the
overall architecture of the project.

For your first patches, this means:

-  be kind, and expect reviewers to be kind in return - LLVM has a `Code
   of Conduct <https://llvm.org/docs/CodeOfConduct.html>`__;

-  be patient - understanding how a new feature fits into the
   architecture of the project is often a time consuming effort, and
   people have to juggle this with other responsibilities in their
   lives; **ping the review once a week** when there is no response;

-  if you can't agree, generally the best way is to do what the reviewer
   asks; we optimize for readability of the code, which the reviewer is
   in a better position to judge; if this feels like it's not the right
   option, you can contact the cfe-dev mailing list to get more feedback
   on the direction;


Commit access
=============

Once you've contributed a handful of patches to LLVM, start to think
about getting commit access yourself. It's probably a good idea if:

-  you've landed 3-5 patches of larger scope than "fix a typo"

-  you'd be willing to review changes that are closely related to yours

-  you'd like to keep contributing to LLVM.


Getting commit access
---------------------

LLVM uses Git for committing changes. The details are in the `developer
policy
document <https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access>`__.


With great power
----------------

Actually, this would be a great time to read the rest of the `developer
policy <https://llvm.org/docs/DeveloperPolicy.html>`__, too. At minimum,
you need to be subscribed to the relevant commits list before landing
changes (e.g. llvm-commits@lists.llvm.org), as discussion often happens
there if a new patch causes problems.


Commit
------

Let's say you have a change on a local git branch, reviewed and ready to
commit. Things to do first:

-  if you used multiple fine-grained commits locally, squash them into a
   single commit. LLVM prefers commits to match the code that was
   reviewed. (If you created one commit and then used "arc diff", you're
   fine)

-  rebase your patch against the latest LLVM code. LLVM uses a linear
   history, so everything should be based on an up-to-date origin/main.

.. code:: console

   $ git pull --rebase https://github.com/llvm/llvm-project.git main

-  ensure the patch looks correct.

.. code:: console

   $ git show

-  run the tests one last time, for good luck

At this point git show should show a single commit on top of
origin/main.

Now you can push your commit with

.. code:: console

   $ git push https://github.com/llvm/llvm-project.git HEAD:main

You should see your change `on
GitHub <https://github.com/llvm/llvm-project/commits/main>`__ within
minutes.


Post-commit errors
------------------

Once your change is submitted it will be picked up by automated build
bots that will build and test your patch in a variety of configurations.

You can see all configurations and their current state in a waterfall
view at http://lab.llvm.org/buildbot/#/waterfall. The waterfall view is good
to get a general overview over the tested configurations and to see
which configuration have been broken for a while.

The console view at http://lab.llvm.org/buildbot/#/console helps to get a
better understanding of the build results of a specific patch. If you
want to follow along how your change is affecting the build bots, **this
should be the first place to look at** - the colored bubbles correspond
to projects in the waterfall.

If you see a broken build, do not despair - some build bots are
continuously broken; if your change broke the build, you will see a red
bubble in the console view, while an already broken build will show an
orange bubble. Of course, even when the build was already broken, a new
change might introduce a hidden new failure.

| When you want to see more details how a specific build is broken,
  click the red bubble.
| If post-commit error logs confuse you, do not worry too much -
  everybody on the project is aware that this is a bit unwieldy, so
  expect people to jump in and help you understand what's going on!

buildbots, overview of bots, getting error logs.


Reverts
-------

if in doubt, revert and re-land.


Conclusion
==========

llvm is a land of contrasts.
