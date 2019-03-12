"""Define text roles for GitHub

* ghissue - Issue
* ghpull - Pull Request
* ghuser - User

Adapted from bitbucket example here:
https://bitbucket.org/birkenfeld/sphinx-contrib/src/tip/bitbucket/sphinxcontrib/bitbucket.py

Authors
-------

* Doug Hellmann
* Min RK
"""
#
# Original Copyright (c) 2010 Doug Hellmann.  All rights reserved.
#

from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes

def make_link_node(rawtext, app, type, slug, options):
    """Create a link to a github resource.

    :param rawtext: Text being replaced with link node.
    :param app: Sphinx application context
    :param type: Link type (issues, changeset, etc.)
    :param slug: ID of the thing to link to
    :param options: Options dictionary passed to role func.
    """

    try:
        base = app.config.github_project_url
        if not base:
            raise AttributeError
        if not base.endswith('/'):
            base += '/'
    except AttributeError as err:
        raise ValueError('github_project_url configuration value is not set (%s)' % str(err))

    ref = base + type + '/' + slug + '/'
    set_classes(options)
    prefix = "#"
    if type == 'pull':
        prefix = "PR " + prefix
    node = nodes.reference(rawtext, prefix + utils.unescape(slug), refuri=ref,
                           **options)
    return node

def ghissue_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Link to a GitHub issue.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """

    try:
        issue_num = int(text)
        if issue_num <= 0:
            raise ValueError
    except ValueError:
        msg = inliner.reporter.error(
            'GitHub issue number must be a number greater than or equal to 1; '
            '"%s" is invalid.' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    app = inliner.document.settings.env.app
    #app.info('issue %r' % text)
    if 'pull' in name.lower():
        category = 'pull'
    elif 'issue' in name.lower():
        category = 'issues'
    else:
        msg = inliner.reporter.error(
            'GitHub roles include "ghpull" and "ghissue", '
            '"%s" is invalid.' % name, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    node = make_link_node(rawtext, app, category, str(issue_num), options)
    return [node], []

def ghuser_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Link to a GitHub user.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    app = inliner.document.settings.env.app
    #app.info('user link %r' % text)
    ref = 'https://www.github.com/' + text
    node = nodes.reference(rawtext, text, refuri=ref, **options)
    return [node], []

def ghcommit_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Link to a GitHub commit.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    app = inliner.document.settings.env.app
    #app.info('user link %r' % text)
    try:
        base = app.config.github_project_url
        if not base:
            raise AttributeError
        if not base.endswith('/'):
            base += '/'
    except AttributeError as err:
        raise ValueError('github_project_url configuration value is not set (%s)' % str(err))

    ref = base + text
    node = nodes.reference(rawtext, text[:6], refuri=ref, **options)
    return [node], []


def setup(app):
    """Install the plugin.
    
    :param app: Sphinx application context.
    """
    app.info('Initializing GitHub plugin')
    app.add_role('ghissue', ghissue_role)
    app.add_role('ghpull', ghissue_role)
    app.add_role('ghuser', ghuser_role)
    app.add_role('ghcommit', ghcommit_role)
    app.add_config_value('github_project_url', None, 'env')
    return
