                       The LLVM Compiler Infrastructure
                           http://llvm.cs.uiuc.edu

Welcome to LLVM!
-----------------
This file is intended to do four things:
(1) help you get started using LLVM;
(2) tell you how to get questions about LLVM answered;
(3) tell you where to find documentation for different kinds of questions; and
(4) tell you about three LLVM-related mailing lists.


Getting Started with LLVM
-------------------------

(1) For license information:
    llvm/LICENSE.txt

(2) Installing and compiling LLVM:
    llvm/docs/GettingStarted.html

(3) Learn about features and limitations of this release:
    llvm/docs/ReleaseNotes.html

(4) Learn how to write a pass within the LLVM system:
    llvm/docs/WritingAnLLVMPass.html 

(5) Learn how to start a new development project using LLVM, where your
    new source code can live anywhere (outside or inside the LLVM tree),
    while using LLVM header files and libraries:
    llvm/docs/Projects.html


Getting Help with LLVM
----------------------

(1) If you have questions or development problems not answered in the
    documentation, send e-mail to llvmdev@cs.uiuc.edu.  This mailing list is
    monitored by all the people in the LLVM group at Illinois, and you should
    expect prompt first responses.

(2) To report a bug, submit a bug report as described in the document:
    http://llvm.cs.uiuc.edu/docs/HowToSubmitABug.html

(3) We now use Bugzilla to track bugs, so you can check the status of
    previous bugs at:
    http://llvm.cs.uiuc.edu/bugs/query.cgi


LLVM Documentation
------------------

All the documents mentioned below except the design overview tech report
are included as part of the LLVM release (in llvm/docs/*):

LLVM Design Overview:
    LLVM : A Compilation Framework for Lifelong Program Analysis
    and Transformation:
        http://llvm.cs.uiuc.edu/pubs/2003-09-30-LifelongOptimizationTR.html

LLVM User Guides:

    Download and Installation Instructions:
        llvm/docs/GettingStarted.html

    LLVM Command Guide:
        llvm/docs/CommandGuide/CommandGuide.html

    LLVM Assembly Language:
        llvm/docs/LangRef.html

    LLVM Test Suite Guide:
        llvm/docs/TestingGuide.html

LLVM Programming Documentation:

    LLVM Programmers Manual:
        llvm/docs/ProgrammersManual.html

    Writing an LLVM Pass:
        llvm/docs/WritingAnLLVMPass.html

    Alias Analysis in LLVM:
        llvm/docs/AliasAnalysis.html

    Command Line Library:
        llvm/docs/CommandLine.html

    Coding Standards:
        llvm/docs/CodingStandards.html

Other LLVM Resources:

    Submitting a Bug:
        http://llvm.cs.uiuc.edu/docs/HowToSubmitABug.html

    Open Projects:
        llvm/docs/OpenProjects.html

    Creating a new LLVM Project:
        llvm/docs/Projects.html

Mailing Lists
--------------
There are three mailing lists for providing LLVM users with information:

(1) LLVM Announcements List:
    http://mail.cs.uiuc.edu/mailman/listinfo/llvm-announce

    This is a low volume list that provides important announcements regarding
    LLVM.  It is primarily intended to announce new releases, major updates to
    the software, etc.  This list is highly recommended for anyone that uses
    LLVM.

(2) LLVM Developers List:
    http://mail.cs.uiuc.edu/mailman/listinfo/llvmdev

    This list is for people who want to be included in technical discussions
    of LLVM.  People post to this list when they have questions about writing
    code for or using the LLVM tools.  It is relatively low volume.

(3) LLVM Commits List
    http://mail.cs.uiuc.edu/mailman/listinfo/llvm-commits

    This list contains all commit messages that are made when LLVM developers
    commit code changes to the CVS archive.  It is useful for those who want to
    stay on the bleeding edge of LLVM development. This list is very high
    volume.

