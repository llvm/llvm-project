.. _phabricator-reviews:

=============================
Code Reviews with Phabricator
=============================

.. contents::
  :local:

If you prefer to use a web user interface for code reviews, you can now submit
your patches for Clang and LLVM at `LLVM's Phabricator`_ instance.

While Phabricator is a useful tool for some, the relevant -commits mailing list
is the system of record for all LLVM code review. The mailing list should be
added as a subscriber on all reviews, and Phabricator users should be prepared
to respond to free-form comments in mail sent to the commits list.

Sign up
-------

To get started with Phabricator, navigate to `https://reviews.llvm.org`_ and
click the power icon in the top right. You can register with a GitHub account,
a Google account, or you can create your own profile.

Make *sure* that the email address registered with Phabricator is subscribed
to the relevant -commits mailing list. If you are not subscribed to the commit
list, all mail sent by Phabricator on your behalf will be held for moderation.

Note that if you use your git user name as Phabricator user name,
Phabricator will automatically connect your submits to your Phabricator user in
the `Code Repository Browser`_.

Requesting a review via the command line
----------------------------------------

Phabricator has a tool called *Arcanist* to upload patches from
the command line. To get you set up, follow the
`Arcanist Quick Start`_ instructions.

You can learn more about how to use arc to interact with
Phabricator in the `Arcanist User Guide`_.
The basic way of creating a revision for the current commit in your local
repository is to run:

::

  arc diff HEAD~


If you later update your commit message, you need to add the `--verbatim`
option to have `arc` update the description on Phabricator:

::

  arc diff --edit --verbatim


.. _phabricator-request-review-web:

Requesting a review via the web interface
-----------------------------------------

The tool to create and review patches in Phabricator is called
*Differential*.

Note that you can upload patches created through git, but using `arc` on the
command line (see previous section) is preferred: it adds more metadata to
Phabricator which are useful for the pre-merge testing system and for
propagating attribution on commits when someone else has to push it for you.

To make reviews easier, please always include **as much context as
possible** with your diff! Don't worry, Phabricator
will automatically send a diff with a smaller context in the review
email, but having the full file in the web interface will help the
reviewer understand your code.

To get a full diff, use one of the following commands (or just use Arcanist
to upload your patch):

* ``git show HEAD -U999999 > mypatch.patch``
* ``git diff -U999999 @{u} > mypatch.patch``
* ``git diff HEAD~1 -U999999 > mypatch.patch``

Before uploading your patch, please make sure it is formatted properly, as
described in :ref:`How to Submit a Patch <format patches>`.

To upload a new patch:

* Click *Differential*.
* Click *+ Create Diff*.
* Paste the text diff or browse to the patch file. Click *Create Diff*.
* Leave this first Repository field blank. (We'll fill in the Repository
  later, when sending the review.)
* Leave the drop down on *Create a new Revision...* and click *Continue*.
* Enter a descriptive title and summary.  The title and summary are usually
  in the form of a :ref:`commit message <commit messages>`.
* Add reviewers (see below for advice). (If you set the Repository field
  correctly, llvm-commits or cfe-commits will be subscribed automatically;
  otherwise, you will have to manually subscribe them.)
* In the Repository field, enter "rG LLVM Github Monorepo".
* Click *Save*.

To submit an updated patch:

* Click *Differential*.
* Click *+ Create Diff*.
* Paste the updated diff or browse to the updated patch file. Click *Create Diff*.
* Select the review you want to from the *Attach To* dropdown and click
  *Continue*.
* Leave the Repository field blank. (We previously filled out the Repository
  for the review request.)
* Add comments about the changes in the new diff. Click *Save*.

Choosing reviewers: You typically pick one or two people as initial reviewers.
This choice is not crucial, because you are merely suggesting and not requiring
them to participate. Many people will see the email notification on cfe-commits
or llvm-commits, and if the subject line suggests the patch is something they
should look at, they will.


.. _finding-potential-reviewers:

Finding potential reviewers
---------------------------

Here are a couple of ways to pick the initial reviewer(s):

* Use ``git blame`` and the commit log to find names of people who have
  recently modified the same area of code that you are modifying.
* Look in CODE_OWNERS.TXT to see who might be responsible for that area.
* If you've discussed the change on a dev list, the people who participated
  might be appropriate reviewers.

Even if you think the code owner is the busiest person in the world, it's still
okay to put them as a reviewer. Being the code owner means they have accepted
responsibility for making sure the review happens.

Reviewing code with Phabricator
-------------------------------

Phabricator allows you to add inline comments as well as overall comments
to a revision. To add an inline comment, select the lines of code you want
to comment on by clicking and dragging the line numbers in the diff pane.
When you have added all your comments, scroll to the bottom of the page and
click the Submit button.

You can add overall comments in the text box at the bottom of the page.
When you're done, click the Submit button.

Phabricator has many useful features, for example allowing you to select
diffs between different versions of the patch as it was reviewed in the
*Revision Update History*. Most features are self descriptive - explore, and
if you have a question, drop by on #llvm in IRC to get help.

Note that as e-mail is the system of reference for code reviews, and some
people prefer it over a web interface, we do not generate automated mail
when a review changes state, for example by clicking "Accept Revision" in
the web interface. Thus, please type LGTM into the comment box to accept
a change from Phabricator.

.. _pre-merge-testing:

Pre-merge testing
-----------------

The pre-merge tests are a continuous integration (CI) workflow. The workflow 
checks the patches uploaded to Phabricator before a user merges them to the main 
branch - thus the term *pre-merge testing*. 

When a user uploads a patch to Phabricator, Phabricator triggers the checks and
then displays the results. This way bugs in a patch are contained during the 
code review stage and do not pollute the main branch. 

If you notice issues or have an idea on how to improve pre-merge checks, please 
`create a new issue <https://github.com/google/llvm-premerge-checks/issues/new>`_ 
or give a ❤️ to an existing one.

Requirements
^^^^^^^^^^^^

To get a patch on Phabricator tested, the build server must be able to apply the
patch to the checked out git repository. Please make sure that either:

* You set a git hash as ``sourceControlBaseRevision`` in Phabricator which is
  available on the GitHub repository, 
* **or** you define the dependencies of your patch in Phabricator, 
* **or** your patch can be applied to the main branch.

Only then can the build server apply the patch locally and run the builds and
tests.

Accessing build results
^^^^^^^^^^^^^^^^^^^^^^^
Phabricator will automatically trigger a build for every new patch you upload or
modify. Phabricator shows the build results at the top of the entry. Clicking on 
the links (in the red box) will show more details:

  .. image:: Phabricator_premerge_results.png

The CI will compile and run tests, run clang-format and clang-tidy on lines
changed.

If a unit test failed, this is shown below the build status. You can also expand
the unit test to see the details:

  .. image:: Phabricator_premerge_unit_tests.png

Committing a change
-------------------

Once a patch has been reviewed and approved on Phabricator it can then be
committed to trunk. If you do not have commit access, someone has to
commit the change for you (with attribution). It is sufficient to add
a comment to the approved review indicating you cannot commit the patch
yourself. If you have commit access, there are multiple workflows to commit the
change. Whichever method you follow it is recommended that your commit message
ends with the line:

::

  Differential Revision: <URL>

where ``<URL>`` is the URL for the code review, starting with
``https://reviews.llvm.org/``.

This allows people reading the version history to see the review for
context. This also allows Phabricator to detect the commit, close the
review, and add a link from the review to the commit.

Note that if you use the Arcanist tool the ``Differential Revision`` line will
be added automatically. If you don't want to use Arcanist, you can add the
``Differential Revision`` line (as the last line) to the commit message
yourself.

Using the Arcanist tool can simplify the process of committing reviewed code as
it will retrieve reviewers, the ``Differential Revision``, etc from the review
and place it in the commit message. You may also commit an accepted change
directly using ``git push``, per the section in the :ref:`getting started
guide <commit_from_git>`.

Note that if you commit the change without using Arcanist and forget to add the
``Differential Revision`` line to your commit message then it is recommended
that you close the review manually. In the web UI, under "Leap Into Action" put
the git revision number in the Comment, set the Action to "Close Revision" and
click Submit.  Note the review must have been Accepted first.

Committing someone's change from Phabricator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a clean Git repository on an up to date ``main`` branch run the
following (where ``<Revision>`` is the Phabricator review number):

::

  arc patch D<Revision>


This will create a new branch called ``arcpatch-D<Revision>`` based on the
current ``main`` and will create a commit corresponding to ``D<Revision>`` with a
commit message derived from information in the Phabricator review.

Check you are happy with the commit message and amend it if necessary.
For example, ensure the 'Author' property of the commit is set to the original author.
You can use a command to correct the author property if it is incorrect:

::

  git commit --amend --author="John Doe <jdoe@llvm.org>"

Then, make sure the commit is up-to-date, and commit it. This can be done by running
the following:

::

  git pull --rebase https://github.com/llvm/llvm-project.git main
  git show # Ensure the patch looks correct.
  ninja check-$whatever # Rerun the appropriate tests if needed.
  git push https://github.com/llvm/llvm-project.git HEAD:main


Abandoning a change
-------------------

If you decide you should not commit the patch, you should explicitly abandon
the review so that reviewers don't think it is still open. In the web UI,
scroll to the bottom of the page where normally you would enter an overall
comment. In the drop-down Action list, which defaults to "Comment," you should
select "Abandon Revision" and then enter a comment explaining why. Click the
Submit button to finish closing the review.

Status
------

Please let us know whether you like it and what could be improved! We're still
working on setting up a bug tracker, but you can email klimek-at-google-dot-com
and chandlerc-at-gmail-dot-com and CC the llvm-dev mailing list with questions
until then. We also could use help implementing improvements. This sadly is
really painful and hard because the Phabricator codebase is in PHP and not as
testable as you might like. However, we've put exactly what we're deploying up
on an `llvm-reviews GitHub project`_ where folks can hack on it and post pull
requests. We're looking into what the right long-term hosting for this is, but
note that it is a derivative of an existing open source project, and so not
trivially a good fit for an official LLVM project.

.. _LLVM's Phabricator: https://reviews.llvm.org
.. _`https://reviews.llvm.org`: https://reviews.llvm.org
.. _Code Repository Browser: https://reviews.llvm.org/diffusion/
.. _Arcanist Quick Start: https://secure.phabricator.com/book/phabricator/article/arcanist_quick_start/
.. _Arcanist User Guide: https://secure.phabricator.com/book/phabricator/article/arcanist/
.. _llvm-reviews GitHub project: https://github.com/r4nt/llvm-reviews/
