# Discourse Migration Guide 

This document is intended to help LLVM users to migrate from the mailing lists to
Discourse. Discourse has two basic ways for interaction: Via the [web
UI](https://llvm.discourse.group/) and via emails.

## Setting up your account

The easiest way is to create an account using your GitHub account:

1. Navigate to https://llvm.discourse.group/
1. Click on "Sign Up" in the top right corner.
1. Choose "With GitHub" on the right side and log in with your GitHub account.

## Structure of Discourse

Discourse's structure is similar to a set of mailing lists, however different
terms are used there. To help with the transition, here's a translation table
for the terms:

<table border=1>
<tr><th>Mailing list</th><th>Discourse</th></tr>
<tr><td><i>Mailing list</i>, consists of threads</td><td><i>category</i>, consists of topics</td></tr>
<tr><td><i>thread</i>, consists of emails</td><td><i>topic</i>, consists of posts</td></tr>
<tr><td>email</td><td>post</td></tr>
</table>

## Setting up email interactions

Some folks want to interact with Discourse purely via their email program. Here
are the typical use cases:

* You can [subscribe to a category or topic](https://discourse.mozilla.org/t/how-do-i-subscribe-to-categories-and-topics/16024)
* You can reply to a post, including quoting other peoples texts
  ([tested](https://llvm.discourse.group/t/email-interaction-with-discourse/3306/4) on GMail).
* [Quoting previous topics in an reply](https://meta.discourse.org/t/single-quote-block-dropped-in-email-reply/144802)
* **TODO:** Creating new topics via email is
  [supported](https://meta.discourse.org/t/start-a-new-topic-via-email/62977)
  but not configured at the moment. We would need to set up an email address
  per category and give Discourse POP3 access to that email account. This sounds
  like a solvable issue.
* You can filter incoming emails in your email client by category using the
  `List-ID` email header field.

## Mapping of mailing lists to categories

This table explains the mapping from mailing lists to categories in Discourse.
The email addresses of these categories will remain the same, after the
migration.  Obsolete lists will become read-only as part of the Discourse
migration.


<table border=1>
<tr><th>Mailing lists</th><th>Category in Discourse</th></tr>

<tr><td>All-commits</td><td>no migration at the moment</td></tr>
<tr><td>Bugs-admin</td><td>no migration at the moment</td></tr>
<tr><td>cfe-commits</td><td>no migration at the moment</td></tr>
<tr><td>cfe-dev</td><td>Clang Frontend</td></tr>
<tr><td>cfe-users</td><td>Clang Frontend/Using Clang</td></tr>
<tr><td>clangd-dev</td><td>Clang Frontend/clangd</td></tr>
<tr><td>devmtg-organizers</td><td>Obsolete</td></tr>
<tr><td>Docs</td><td>Obsolete</td></tr>
<tr><td>eurollvm-organizers</td><td>Obsolete</td></tr>
<tr><td>flang-commits</td><td>no migration at the moment</td></tr>
<tr><td>flang-dev</td><td>Subprojects/Flang Fortran Frontend</td></tr>
<tr><td>gsoc</td><td>Obsolete</td></tr>
<tr><td>libc-commits</td><td>no migration at the moment</td></tr>
<tr><td>libc-dev</td><td>Runtimes/C</td></tr>
<tr><td>Libclc-dev</td><td>Runtimes/OpenCL</td></tr>
<tr><td>libcxx-bugs</td><td>no migration at the moment</td></tr>
<tr><td>libcxx-commits</td><td>no migration at the moment</td></tr>
<tr><td>libcxx-dev</td><td>Runtimes/C++</td></tr>
<tr><td>lldb-commits</td><td>no migration at the moment</td></tr>
<tr><td>lldb-dev</td><td>Subprojects/lldb</td></tr>
<tr><td>llvm-admin</td><td>no migration at the moment</td></tr>
<tr><td>llvm-announce</td><td>Announce</td></tr>
<tr><td>llvm-branch-commits</td><td>no migration at the moment</td></tr>
<tr><td>llvm-bugs</td><td>no migration at the moment</td></tr>
<tr><td>llvm-commits</td><td>no migration at the moment</td></tr>
<tr><td>llvm-dev</td><td>Project Infrastructure/LLVM Dev List Archives</td></tr>
<tr><td>llvm-devmeeting</td><td>Community/US Developer Meeting</td></tr>
<tr><td>llvm-foundation</td><td>Community/LLVM Foundation</td></tr>
<tr><td>Mlir-commits</td><td>no migration at the moment</td></tr>
<tr><td>Openmp-commits</td><td>no migration at the moment</td></tr>
<tr><td>Openmp-dev</td><td>Runtimes/OpenMP</td></tr>
<tr><td>Parallel_libs-commits</td><td>no migration at the moment</td></tr>
<tr><td>Parallel_libs-dev</td><td>Runtimes/C++</td></tr>
<tr><td>Release-testers</td><td>Project Infrastructure/Release Testers</td></tr>
<tr><td>Test-list</td><td>Obsolete</td></tr>
<tr><td>vmkit-commits</td><td>Obsolete</td></tr>
<tr><td>WiCT</td><td>Community/Women in Compilers and Tools</td></tr>
<tr><td>www-scripts</td><td>Obsolete</td></tr> 
</table>


## FAQ

### I don't want to use a web UI

You can do most of the communication with your email client (see section on
Setting up email interactions above). You only need to set up your account once
and then configure which categories you want to subscribe to.

### How do I send a private message?

On the mailing list you have the opportunity to reply only to the sender of
the email, not to the entire list. That is not supported when replying via
email on Discourse. However you can send someone a private message via the
Web UI: Click on the user's name above a post and then on `Message`.

Also Discourse does not expose users' email addresses , so your private
replies have to go through their platform (unless you happen to know the
email address of the user.)

### How can my script/tool send automatic messages?**

In case you want to [create a new
post/topic](https://docs.discourse.org/#tag/Posts/paths/~1posts.json/post)
automatically from a script or tool, you can use the
[Discourse API](https://docs.discourse.org/).

### Who are the admins for Discourse?

See https://llvm.discourse.group/about

### What is the reason for the migration?

See
[this email](https://lists.llvm.org/pipermail/llvm-dev/2021-June/150823.html)

### How do I set up a private mailing list?

If needed categories can have individual [security
settings](https://meta.discourse.org/t/how-to-use-category-security-settings-to-create-private-categories/87678)
to limit visibility and write permissions. Contact the
[admins](https://llvm.discourse.group/about) if you need such a category.

### What will happen to our email archives?

The Mailman archives will remain on the web server for now.

### What are advantages of Discourse over the current mailing lists?

* Users can post to any category, also without being subscribed.
* Full text search on the Web UI.
* Sending/replying via the Web UI (email is still possible).
* View entire thread on one page.
* Categories are a more light-weight option to structure the discussions than
  creating new mailing lists.
* Single sign on with GitHub.
* User email addresses are kept private.

### I have another question not covered here. What should I do?

Please contact iwg@llvm.org or raise a
[ticket on GitHub](https://github.com/llvm/llvm-iwg/issues).
