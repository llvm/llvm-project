================
LLVM Admin Tasks
================

Commit Access Review
--------------------

* Frequency: Monthly
* Permissions: Triage
* Description: The `Commit Access Review <https://github.com/llvm/llvm-project/actions/workflows/commit-access-review.yml>`_
  github actions job runs once per month and generates an artifact called 'triagers' which is a list of GitHub users who
  no longer qualify for commit access due to limited activity over the last 12 months.

When the job completes, an admin should download the triagers artifact and create an issue asking everyone on
the list if they still need commit access.  Here is an `example issue <https://github.com/llvm/llvm-project/issues/131262>`_.
The issue should have the 'infra:commit-access' label and should use the same description as the example issue.

Removing Users from LLVM Committers team
----------------------------------------

* Frequencey: Monthly
* Permissions: Admin
* Description: Each month an admin should review any open issues with the infra:commit-access label.
  For any issue that has been open for more than 4 weeks, any user who has not responded on the ticket
  should be moved from the LLVM Committers team to the LLVM Triagers team.

Action Secret Rotation
----------------------

* Frequency: Monthly
* Permissions: Admin
* Description: Each month an admin should rotate the secrets that are used for GitHub Actions workflows
  in the llvm-project repository.

The secrets are personal access tokens that are associated with the llvmbot GitHub account.  An admin
should log in to the llvmbot account, re-generate the personal access tokens and then copy the
new values into the corresponding secret in the llvm-project repo's settings.

Grant Commit Access
--------------------

* Frequency: Ongoing
* Permissions: Admin
* Description: An admin should periodically review the list of commit access requests.  This can be
  done by searching for issues in the llvm-project repository which have the infra:commit-access-request
  label.  Any user that meets the `commit access requirements <https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access>`_
  should be added to the 'LLVM Committers' team.  The admin should add a comment to the issue when
  an invite to join the team has been sent and then close the issue when the invite has been accepted.
